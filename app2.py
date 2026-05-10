import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pulp

# Page configuration
st.set_page_config(
    page_title="🌿 원예장비 총괄생산계획 시스템",
    layout="wide",
    page_icon="🌿"
)

# Default parameters
DEFAULT_PARAMS = {
    'demand': [1600, 3000, 3200, 3800, 2200, 2200],
    'p': 40,   # Selling price
    'm': 10,   # Material cost
    'h': 2,    # Inventory holding cost
    'b': 5,    # Backlog cost
    'I0': 1000,
    'I6': 500,
    'W0': 80,
    'cr': 4,
    'co': 6,
    'ch': 300,
    'cf': 500,
    'd': 20,
    'r': 8,
    'OT_max': 10,
    's': 4
}

if 'params' not in st.session_state:
    st.session_state.params = DEFAULT_PARAMS.copy()
# reset_key: 초기화 버튼을 누를 때마다 1씩 증가 → 위젯 key가 바뀌어 강제 재생성
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# ── Sidebar ──────────────────────────────────────────
st.sidebar.title("📋 파라미터 설정")

# ✅ rk를 number_input보다 먼저 선언 (NameError 수정)
rk = st.session_state.reset_key

if st.sidebar.button("🔄 초기화"):
    st.session_state.params = DEFAULT_PARAMS.copy()
    st.session_state.reset_key += 1
    st.rerun()

st.sidebar.subheader("수요 예측 (Jan–Jun)")
demand = []
for i, month in enumerate(['1월', '2월', '3월', '4월', '5월', '6월']):
    demand.append(st.sidebar.number_input(f"{month} 수요", value=st.session_state.params['demand'][i], min_value=0, step=100, key=f"demand_{i}_{rk}"))
st.session_state.params['demand'] = demand

st.sidebar.subheader("비용 및 용량 파라미터")
cols = st.sidebar.columns(2)
with cols[0]:
    st.session_state.params['p']   = st.sidebar.number_input("판매가격 (천원/단위)",      value=float(st.session_state.params['p']),   min_value=0.0, step=1.0,  key=f"p_{rk}")
    st.session_state.params['m']   = st.sidebar.number_input("재료비 (천원/단위)",         value=float(st.session_state.params['m']),   min_value=0.0, step=1.0,  key=f"m_{rk}")
    st.session_state.params['h']   = st.sidebar.number_input("재고보유비 (천원/단위/월)",  value=float(st.session_state.params['h']),   min_value=0.0, step=0.1,  key=f"h_{rk}")
    st.session_state.params['b']   = st.sidebar.number_input("부재비 (천원/단위/월)",      value=float(st.session_state.params['b']),   min_value=0.0, step=0.1,  key=f"b_{rk}")
    st.session_state.params['I0']  = st.sidebar.number_input("초기재고 (단위)",            value=st.session_state.params['I0'],         min_value=0,   step=100,  key=f"I0_{rk}")
    st.session_state.params['I6']  = st.sidebar.number_input("최종재고 (단위)",            value=st.session_state.params['I6'],         min_value=0,   step=100,  key=f"I6_{rk}")
with cols[1]:
    st.session_state.params['W0']     = st.sidebar.number_input("초기인력 (명)",            value=st.session_state.params['W0'],            min_value=0,   step=1,    key=f"W0_{rk}")
    st.session_state.params['cr']     = st.sidebar.number_input("정규임금 (천원/시간)",     value=float(st.session_state.params['cr']),     min_value=0.0, step=0.1,  key=f"cr_{rk}")
    st.session_state.params['co']     = st.sidebar.number_input("초과임금 (천원/시간)",     value=float(st.session_state.params['co']),     min_value=0.0, step=0.1,  key=f"co_{rk}")
    st.session_state.params['ch']     = st.sidebar.number_input("채용비 (천원/명)",         value=st.session_state.params['ch'],            min_value=0,   step=10,   key=f"ch_{rk}")
    st.session_state.params['cf']     = st.sidebar.number_input("해고비 (천원/명)",         value=st.session_state.params['cf'],            min_value=0,   step=10,   key=f"cf_{rk}")
    st.session_state.params['d']      = st.sidebar.number_input("월근무일수 (일)",          value=st.session_state.params['d'],             min_value=1,   step=1,    key=f"d_{rk}")
    st.session_state.params['r']      = st.sidebar.number_input("일정규시간 (시간)",        value=float(st.session_state.params['r']),      min_value=1.0, step=0.5,  key=f"r_{rk}")
    st.session_state.params['OT_max'] = st.sidebar.number_input("최대초과시간 (시간/명/월)",value=float(st.session_state.params['OT_max']), min_value=0.0, step=1.0,  key=f"OT_max_{rk}")
    st.session_state.params['s']      = st.sidebar.number_input("단위생산시간 (시간/단위)", value=float(st.session_state.params['s']),      min_value=0.1, step=0.1,  key=f"s_{rk}")


all_strategies = ["Level Production (평준화)", "Chase Demand (추종)", "Mixed (최적화)", "Overtime-Only (초과근무)"]
# ✅ Fix: selected_strategies를 함수 인자로 전달해 캐시가 올바르게 무효화되도록 수정
selected_strategies = st.sidebar.multiselect("전략 선택", all_strategies, default=all_strategies)

# ── Main ─────────────────────────────────────────────
st.title("🌿 원예장비 총괄생산계획 시스템")


@st.cache_data
def compute_strategies(params, selected):
    D = np.array(params['demand'])
    T = len(D)
    results = {}

    # ── Strategy 1: Level Production ──────────────────
    if "Level Production (평준화)" in selected:
        W = np.full(T, params['W0'])
        cap_reg = W * params['d'] * params['r'] / params['s']
        P = np.zeros(T); O = np.zeros(T)
        I = np.zeros(T + 1); B = np.zeros(T + 1)
        I[0] = params['I0']

        for t in range(T):
            required = D[t] + (params['I6'] - I[t]) / (T - t) if t < T - 1 else D[t] + (params['I6'] - I[t])
            P[t] = min(cap_reg[t], required)
            if P[t] < required:
                ot_needed = (required - P[t]) * params['s']
                O[t] = min(ot_needed, W[t] * params['OT_max'])
                P[t] += O[t] / params['s']
            I[t + 1] = I[t] + P[t] - D[t]
            if I[t + 1] < 0:
                B[t + 1] = -I[t + 1]
                I[t + 1] = 0

        results["Level Production (평준화)"] = {
            'P': P, 'W': W, 'H': np.zeros(T), 'F': np.zeros(T),
            'O': O, 'I': I[1:], 'B': B[1:]
        }

    # ── Strategy 2: Chase Demand ───────────────────────
    if "Chase Demand (추종)" in selected:
        W = np.zeros(T); P = np.zeros(T); O = np.zeros(T)
        I = np.zeros(T + 1); B = np.zeros(T + 1)
        H = np.zeros(T); F = np.zeros(T)
        I[0] = params['I0']; W_prev = params['W0']

        for t in range(T):
            target = D[t] + (params['I6'] - I[t])
            req_w = int(np.ceil(target * params['s'] / (params['d'] * params['r'])))
            H[t] = max(0, req_w - W_prev)
            F[t] = max(0, W_prev - req_w)
            W[t] = W_prev + H[t] - F[t]
            W_prev = W[t]

            cap = W[t] * params['d'] * params['r'] / params['s']
            P[t] = min(cap, target)
            if P[t] < target:
                ot_needed = (target - P[t]) * params['s']
                O[t] = min(ot_needed, W[t] * params['OT_max'])
                P[t] += O[t] / params['s']

            I[t + 1] = I[t] + P[t] - D[t]
            if I[t + 1] < 0:
                B[t + 1] = -I[t + 1]
                I[t + 1] = 0

        results["Chase Demand (추종)"] = {
            'P': P, 'W': W, 'H': H, 'F': F, 'O': O, 'I': I[1:], 'B': B[1:]
        }

    # ── Strategy 3: Mixed (MILP) ───────────────────────
    # ✅ Fix: 부재 변수(B) 제거 + 재고균형을 I_t = I_{t-1} + P_t - D_t 로 단순화
    # I_v의 lowBound=0 이 I >= 0 을 강제 → 부재를 허용하지 않음
    # 기존 코드는 부재비(5)가 노동비(640/인)보다 훨씬 싸서
    # LP가 "사람 다 해고 + 부재 무한 누적"을 최적해로 선택하는 버그 발생
    if "Mixed (최적화)" in selected:
        prob = pulp.LpProblem("APP_MILP", pulp.LpMinimize)

        P_v = [pulp.LpVariable(f"P_{t}", lowBound=0)                for t in range(T)]
        W_v = [pulp.LpVariable(f"W_{t}", lowBound=0, cat='Integer') for t in range(T)]
        H_v = [pulp.LpVariable(f"H_{t}", lowBound=0, cat='Integer') for t in range(T)]
        F_v = [pulp.LpVariable(f"F_{t}", lowBound=0, cat='Integer') for t in range(T)]
        O_v = [pulp.LpVariable(f"O_{t}", lowBound=0)                for t in range(T)]
        I_v = [pulp.LpVariable(f"I_{t}", lowBound=0)                for t in range(T)]

        # 목적함수 (부재 항 없음 — 부재 자체를 허용하지 않으므로)
        prob += pulp.lpSum(
            params['m']  * P_v[t]
            + params['d'] * params['r'] * params['cr'] * W_v[t]
            + params['ch'] * H_v[t]
            + params['cf'] * F_v[t]
            + params['co'] * O_v[t]
            + params['h']  * I_v[t]
            for t in range(T)
        )

        # 재고 균형 (I_v >= 0 이 부재 방지, 용량 부족 시 모델이 infeasible)
        for t in range(T):
            prev_I = params['I0'] if t == 0 else I_v[t - 1]
            prob += I_v[t] == prev_I + P_v[t] - D[t]

        # 인력 균형
        for t in range(T):
            prev_W = params['W0'] if t == 0 else W_v[t - 1]
            prob += W_v[t] == prev_W + H_v[t] - F_v[t]

        # 생산 용량 제약
        for t in range(T):
            prob += P_v[t] * params['s'] <= W_v[t] * params['d'] * params['r'] + O_v[t]

        # 초과근무 한도
        for t in range(T):
            prob += O_v[t] <= W_v[t] * params['OT_max']

        # 기말재고 보장
        prob += I_v[T - 1] >= params['I6']

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if status == 1:
            results["Mixed (최적화)"] = {
                'P': np.array([pulp.value(v) for v in P_v]),
                'W': np.array([pulp.value(v) for v in W_v]),
                'H': np.array([pulp.value(v) for v in H_v]),
                'F': np.array([pulp.value(v) for v in F_v]),
                'O': np.array([pulp.value(v) for v in O_v]),
                'I': np.array([pulp.value(v) for v in I_v]),
                'B': np.zeros(T),
            }
        else:
            st.error("⚠️ Mixed 최적화 실패 — 현재 인력·용량으로 수요를 충족할 수 없습니다. 파라미터를 확인하세요.")
            results["Mixed (최적화)"] = None

    # ── Strategy 4: Overtime-Only ──────────────────────
    if "Overtime-Only (초과근무)" in selected:
        W = np.full(T, params['W0'])
        cap_reg = W * params['d'] * params['r'] / params['s']
        max_ot_prod = W * params['OT_max'] / params['s']
        P = cap_reg + max_ot_prod
        O = W * params['OT_max']
        I = np.zeros(T + 1); B = np.zeros(T + 1)
        I[0] = params['I0']

        for t in range(T):
            I[t + 1] = I[t] + P[t] - D[t]
            if I[t + 1] < 0:
                B[t + 1] = -I[t + 1]
                I[t + 1] = 0

        results["Overtime-Only (초과근무)"] = {
            'P': P, 'W': W, 'H': np.zeros(T), 'F': np.zeros(T),
            'O': O, 'I': I[1:], 'B': B[1:]
        }

    return results


results = compute_strategies(st.session_state.params, tuple(selected_strategies))


def compute_costs(data, params):
    P, W, H, F, O, I, B = data['P'], data['W'], data['H'], data['F'], data['O'], data['I'], data['B']
    reg_labor = W * params['d'] * params['r'] * params['cr']
    ot_cost   = O * params['co']
    hire_cost = H * params['ch']
    fire_cost = F * params['cf']
    inv_cost  = I * params['h']
    back_cost = B * params['b']
    mat_cost  = P * params['m']
    monthly   = reg_labor + ot_cost + hire_cost + fire_cost + inv_cost + back_cost + mat_cost
    return {
        'monthly': monthly,
        'total': float(np.sum(monthly)),
        'breakdown': {
            '정규노동비': float(np.sum(reg_labor)),
            '초과근무비': float(np.sum(ot_cost)),
            '채용비':     float(np.sum(hire_cost)),
            '해고비':     float(np.sum(fire_cost)),
            '재고보유비': float(np.sum(inv_cost)),
            '부재비':     float(np.sum(back_cost)),
            '재료비':     float(np.sum(mat_cost)),
        }
    }


months = ['1월', '2월', '3월', '4월', '5월', '6월']

tab1, tab2, tab3, tab4 = st.tabs(["📊 전략 비교 요약", "📅 월별 상세 계획", "📈 시각화 대시보드", "💡 계획 평가 및 권고"])

# ── Tab 1: Summary ────────────────────────────────────
with tab1:
    st.header("📊 전략 비교 요약")

    summary_data = []
    for strategy, data in results.items():
        if data is None:
            continue
        costs = compute_costs(data, st.session_state.params)
        summary_data.append({
            '전략':        strategy,
            '총비용':      costs['total'],
            '총생산':      float(np.sum(data['P'])),
            '평균인력':    float(np.mean(data['W'])),
            '총초과시간':  float(np.sum(data['O'])),
            '총재고':      float(np.sum(data['I'])),
            '총부재':      float(np.sum(data['B'])),
            '서비스레벨%': (len(data['B']) - np.count_nonzero(data['B'])) / len(data['B']) * 100,
        })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.format({
        '총비용':      '{:,.0f}',
        '총생산':      '{:,.0f}',
        '평균인력':    '{:.1f}',
        '총초과시간':  '{:.1f}',
        '총재고':      '{:,.0f}',
        '총부재':      '{:,.0f}',
        '서비스레벨%': '{:.1f}%',
    }))

    fig = px.bar(df_summary, x='전략', y='총비용', title='전략별 총비용 비교', color='전략')
    st.plotly_chart(fig, use_container_width=True)

    min_idx = df_summary['총비용'].idxmin()
    st.success(f"🏆 최저비용 전략: {df_summary.loc[min_idx, '전략']} "
               f"(총비용: {df_summary.loc[min_idx, '총비용']:,.0f} 천원)")

# ── Tab 2: Monthly Detail ────────────────────────────
with tab2:
    st.header("📅 월별 상세 계획")

    valid_keys = [k for k, v in results.items() if v is not None]
    strategy_select = st.selectbox("전략 선택", valid_keys)

    if strategy_select and results[strategy_select] is not None:
        data = results[strategy_select]
        costs = compute_costs(data, st.session_state.params)

        df_detail = pd.DataFrame({
            '월':          months,
            '수요':        st.session_state.params['demand'],
            '생산':        data['P'],
            '인력':        data['W'],
            '채용':        data['H'],
            '해고':        data['F'],
            '초과시간(h)': data['O'],
            '기말재고':    data['I'],
            '부재':        data['B'],
            '월비용':      costs['monthly'],
        })

        def highlight_backlog(val):
            return 'background-color: #ffcccc; color: red; font-weight: bold;' if val > 0 else ''

        st.dataframe(df_detail.style
            .format({
                '수요':        '{:,.0f}',
                '생산':        '{:,.1f}',
                '인력':        '{:.0f}',
                '채용':        '{:.0f}',
                '해고':        '{:.0f}',
                '초과시간(h)': '{:.1f}',
                '기말재고':    '{:,.1f}',
                '부재':        '{:,.1f}',
                '월비용':      '{:,.0f}',
            })
            .map(highlight_backlog, subset=['부재'])
        )

# ── Tab 3: Charts ────────────────────────────────────
with tab3:
    st.header("📈 시각화 대시보드")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=months, y=st.session_state.params['demand'], name='수요', marker_color='steelblue'))
        for s, d in results.items():
            if d is not None:
                fig1.add_trace(go.Bar(x=months, y=d['P'], name=f'{s} 생산', opacity=0.75))
        fig1.update_layout(title='수요 vs 생산량', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for s, d in results.items():
            if d is not None:
                fig2.add_trace(go.Scatter(x=months, y=d['I'], mode='lines+markers', name=f'{s} 재고'))
                if np.sum(d['B']) > 0:
                    fig2.add_trace(go.Scatter(x=months, y=d['B'], mode='lines',
                                              line=dict(dash='dot'), name=f'{s} 부재'))
        fig2.add_hline(y=st.session_state.params['I6'], line_dash='dot',
                       annotation_text=f"목표 기말재고: {st.session_state.params['I6']}")
        fig2.update_layout(title='재고 및 부재 추이')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = go.Figure()
        for s, d in results.items():
            if d is not None:
                fig3.add_trace(go.Scatter(x=months, y=d['W'], mode='lines+markers', name=f'{s} 인력'))
        fig3.add_hline(y=st.session_state.params['W0'], line_dash='dot',
                       annotation_text=f"초기 인력: {st.session_state.params['W0']}")
        fig3.update_layout(title='월별 인력 변동')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cost_rows = []
        for s, d in results.items():
            if d is not None:
                for ctype, amt in compute_costs(d, st.session_state.params)['breakdown'].items():
                    cost_rows.append({'전략': s, '비용유형': ctype, '금액': amt})
        fig4 = px.bar(pd.DataFrame(cost_rows), x='전략', y='금액',
                      color='비용유형', title='비용 구성', barmode='stack')
        st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    for s, d in results.items():
        if d is not None:
            cum = np.cumsum(compute_costs(d, st.session_state.params)['monthly'])
            fig5.add_trace(go.Scatter(x=months, y=cum, mode='lines+markers', name=s))
    fig5.update_layout(title='누적 비용 비교')
    st.plotly_chart(fig5, use_container_width=True)

    sl_rows = [{'전략': s,
                '서비스레벨%': (len(d['B']) - np.count_nonzero(d['B'])) / len(d['B']) * 100}
               for s, d in results.items() if d is not None]
    fig6 = px.bar(pd.DataFrame(sl_rows), x='전략', y='서비스레벨%',
                  title='서비스 레벨', range_y=[0, 110])
    st.plotly_chart(fig6, use_container_width=True)

# ── Tab 4: Evaluation ────────────────────────────────
with tab4:
    st.header("💡 계획 평가 및 권고")

    if summary_data:
        df_sum = pd.DataFrame(summary_data)
        best = df_sum.loc[df_sum['총비용'].idxmin()]
        worst_cost = df_sum['총비용'].max()
        savings = worst_cost - best['총비용']
        savings_pct = savings / worst_cost * 100 if worst_cost > 0 else 0

        st.write(f"**최저비용 전략:** {best['전략']} (총비용: {best['총비용']:,.0f} 천원)")
        st.write(f"**최대 절감 가능액:** {savings:,.0f} 천원 ({savings_pct:.1f}%)")

        backlog_st = [s for s, d in results.items() if d is not None and np.sum(d['B']) > 0]
        if backlog_st:
            st.warning(f"⚠️ 부재 발생 전략: {', '.join(backlog_st)} — 고객 서비스 리스크 있음")
        else:
            st.success("✅ 모든 전략에서 부재 없음")

        ot_limit_st = [s for s, d in results.items()
                       if d is not None and np.any(d['O'] >= d['W'] * st.session_state.params['OT_max'])]
        if ot_limit_st:
            st.warning(f"⚠️ 초과근무 한계 도달 전략: {', '.join(ot_limit_st)}")

        unstable_st = [s for s, d in results.items()
                       if d is not None and (np.sum(d['H']) + np.sum(d['F'])) > 0]
        if unstable_st:
            st.info(f"ℹ️ 인력 변동 전략: {', '.join(unstable_st)} — 인력 안정성 고려 필요")

        st.subheader("📋 최종 권고")
        rec = best['전략']
        if rec == "Mixed (최적화)":
            st.success("혼합 최적화 전략을 권장합니다. 수학적으로 검증된 최소 비용 솔루션입니다.")
        elif rec == "Level Production (평준화)":
            st.info("평준화 전략을 권장합니다. 인력 안정성과 예측 가능성이 높습니다.")
        elif rec == "Chase Demand (추종)":
            st.info("추종 전략을 고려해보세요. 수요 변동에 민첩하게 대응할 수 있습니다.")
        else:
            st.info("초과근무 전략은 단기 대응에 적합하나 장기적으로는 비용이 높습니다.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최저비용 전략", best['전략'])
        with col2:
            st.metric("총비용 절감액", f"{savings:,.0f} 천원")
        with col3:
            max_inv = max(float(np.max(d['I'])) for d in results.values() if d is not None)
            st.metric("최대재고", f"{max_inv:,.0f} 단위")
        with col4:
            total_back = sum(float(np.sum(d['B'])) for d in results.values() if d is not None)
            st.metric("총부재", f"{total_back:,.0f} 단위")

# ── Formulas Expander ─────────────────────────────────
with st.expander("📐 수식 및 계산 로직 보기"):
    st.markdown("""
### 주요 수식

**재고 균형 (Mixed 전략 — 부재 없음):**
```
I_t = I_{t-1} + P_t - D_t   (I_t ≥ 0 강제)
```

**재고 균형 (휴리스틱 전략 — 부재 허용):**
```
I_t = max(0, I_{t-1} + P_t - D_t)
B_t = max(0, D_t - I_{t-1} - P_t)
```

**인력 균형:**
```
W_t = W_{t-1} + H_t - F_t
```

**정규 용량 제약:**
```
P_t × s ≤ W_t × d × r + O_t
```

**초과근무 한계:**
```
O_t ≤ W_t × OT_max
```

**월별 비용:**
```
비용 = 정규노동비 + 초과노동비 + 채용비 + 해고비 + 재고보유비 + 부재비 + 재료비
```

**서비스 레벨:**
```
서비스레벨% = (부재 없는 월 수 / 총 월 수) × 100
```
    """)
