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
    'p': 40,  # Selling price
    'm': 10,  # Material cost
    'h': 2,   # Inventory holding cost
    'b': 5,   # Backlog cost
    'I0': 1000,  # Initial inventory
    'I6': 500,   # Required ending inventory
    'W0': 80,    # Current workforce
    'cr': 4,     # Regular wage rate
    'co': 6,     # Overtime wage rate
    'ch': 300,   # Hiring cost
    'cf': 500,   # Firing cost
    'd': 20,     # Working days per month
    'r': 8,      # Regular hours per day
    'OT_max': 10,  # Max overtime per worker/mo
    's': 4       # Standard labor hours/unit
}

# Initialize session state
if 'params' not in st.session_state:
    st.session_state.params = DEFAULT_PARAMS.copy()

# Sidebar
st.sidebar.title("📋 파라미터 설정")

# Demand inputs
st.sidebar.subheader("수요 예측 (Jan–Jun)")
demand = []
for i, month in enumerate(['1월', '2월', '3월', '4월', '5월', '6월']):
    demand.append(st.sidebar.number_input(f"{month} 수요", value=st.session_state.params['demand'][i], min_value=0, step=100))
st.session_state.params['demand'] = demand

# Other parameters
st.sidebar.subheader("비용 및 용량 파라미터")
cols = st.sidebar.columns(2)
with cols[0]:
    st.session_state.params['p'] = st.sidebar.number_input("판매가격 (천원/단위)", value=float(st.session_state.params['p']), min_value=0.0, step=1.0)
    st.session_state.params['m'] = st.sidebar.number_input("재료비 (천원/단위)", value=float(st.session_state.params['m']), min_value=0.0, step=1.0)
    st.session_state.params['h'] = st.sidebar.number_input("재고보유비 (천원/단위/월)", value=float(st.session_state.params['h']), min_value=0.0, step=0.1)
    st.session_state.params['b'] = st.sidebar.number_input("부재비 (천원/단위/월)", value=float(st.session_state.params['b']), min_value=0.0, step=0.1)
    st.session_state.params['I0'] = st.sidebar.number_input("초기재고 (단위)", value=st.session_state.params['I0'], min_value=0, step=100)
    st.session_state.params['I6'] = st.sidebar.number_input("최종재고 (단위)", value=st.session_state.params['I6'], min_value=0, step=100)

with cols[1]:
    st.session_state.params['W0'] = st.sidebar.number_input("초기인력 (명)", value=st.session_state.params['W0'], min_value=0, step=1)
    st.session_state.params['cr'] = st.sidebar.number_input("정규임금 (천원/시간)", value=float(st.session_state.params['cr']), min_value=0.0, step=0.1)
    st.session_state.params['co'] = st.sidebar.number_input("초과임금 (천원/시간)", value=float(st.session_state.params['co']), min_value=0.0, step=0.1)
    st.session_state.params['ch'] = st.sidebar.number_input("채용비 (천원/명)", value=st.session_state.params['ch'], min_value=0, step=10)
    st.session_state.params['cf'] = st.sidebar.number_input("해고비 (천원/명)", value=st.session_state.params['cf'], min_value=0, step=10)
    st.session_state.params['d'] = st.sidebar.number_input("월근무일수 (일)", value=st.session_state.params['d'], min_value=1, step=1)
    st.session_state.params['r'] = st.sidebar.number_input("일정규시간 (시간)", value=float(st.session_state.params['r']), min_value=1.0, step=0.5)
    st.session_state.params['OT_max'] = st.sidebar.number_input("최대초과시간 (시간/명/월)", value=float(st.session_state.params['OT_max']), min_value=0.0, step=1.0)
    st.session_state.params['s'] = st.sidebar.number_input("단위생산시간 (시간/단위)", value=float(st.session_state.params['s']), min_value=0.1, step=0.1)

# Reset button
if st.sidebar.button("🔄 초기화"):
    st.session_state.params = DEFAULT_PARAMS.copy()
    st.rerun()

# Strategy selector
strategies = ["Level Production (평준화)", "Chase Demand (추종)", "Mixed (최적화)", "Overtime-Only (초과근무)"]
selected_strategies = st.sidebar.multiselect("전략 선택", strategies, default=strategies)

# Main area
st.title("🌿 원예장비 총괄생산계획 시스템")

# Function to compute strategies
@st.cache_data
def compute_strategies(params):
    D = np.array(params['demand'])
    T = len(D)
    
    results = {}
    
    # Strategy 1: Level Production
    if "Level Production (평준화)" in selected_strategies:
        W = np.full(T, params['W0'])
        capacity_regular = W * params['d'] * params['r'] / params['s']
        P = np.zeros(T)
        O = np.zeros(T)
        I = np.zeros(T+1)
        B = np.zeros(T+1)
        I[0] = params['I0']
        B[0] = 0
        
        for t in range(T):
            # Use overtime to meet demand if needed
            required = D[t] + (params['I6'] - I[t]) / (T - t) if t < T-1 else D[t] + (params['I6'] - I[t])
            P[t] = min(capacity_regular[t], required)
            if P[t] < required:
                overtime_needed = (required - P[t]) * params['s']
                O[t] = min(overtime_needed, W[t] * params['OT_max'])
                P[t] += O[t] / params['s']
            
            I[t+1] = I[t] + P[t] - D[t]
            if I[t+1] < 0:
                B[t+1] = -I[t+1]
                I[t+1] = 0
        
        H = np.zeros(T)
        F = np.zeros(T)
        results["Level Production (평준화)"] = {'P': P, 'W': W, 'H': H, 'F': F, 'O': O, 'I': I[1:], 'B': B[1:]}
    
    # Strategy 2: Chase Demand
    if "Chase Demand (추종)" in selected_strategies:
        W = np.zeros(T)
        P = np.zeros(T)
        O = np.zeros(T)
        I = np.zeros(T+1)
        B = np.zeros(T+1)
        H = np.zeros(T)
        F = np.zeros(T)
        I[0] = params['I0']
        B[0] = 0
        W[0] = params['W0']
        
        for t in range(T):
            target_production = D[t]
            if t == 0:
                target_production += params['I6'] - I[0]  # Adjust for ending inventory
            else:
                target_production += params['I6'] - I[t]
            
            required_workers = np.ceil(target_production * params['s'] / (params['d'] * params['r']))
            if t > 0:
                H[t] = max(0, required_workers - W[t-1])
                F[t] = max(0, W[t-1] - required_workers)
                W[t] = W[t-1] + H[t] - F[t]
            else:
                W[t] = required_workers
            
            capacity = W[t] * params['d'] * params['r'] / params['s']
            P[t] = min(capacity, target_production)
            if P[t] < target_production:
                overtime_needed = (target_production - P[t]) * params['s']
                O[t] = min(overtime_needed, W[t] * params['OT_max'])
                P[t] += O[t] / params['s']
            
            I[t+1] = I[t] + P[t] - D[t]
            if I[t+1] < 0:
                B[t+1] = -I[t+1]
                I[t+1] = 0
        
        results["Chase Demand (추종)"] = {'P': P, 'W': W, 'H': H, 'F': F, 'O': O, 'I': I[1:], 'B': B[1:]}
    
    # Strategy 3: Mixed (LP)
    if "Mixed (최적화)" in selected_strategies:
        # Create LP problem
        prob = pulp.LpProblem("APP_Optimization", pulp.LpMinimize)
        
        # Variables
        P_vars = [pulp.LpVariable(f"P_{t+1}", lowBound=0) for t in range(T)]
        W_vars = [pulp.LpVariable(f"W_{t+1}", lowBound=0, cat='Integer') for t in range(T)]
        H_vars = [pulp.LpVariable(f"H_{t+1}", lowBound=0, cat='Integer') for t in range(T)]
        F_vars = [pulp.LpVariable(f"F_{t+1}", lowBound=0, cat='Integer') for t in range(T)]
        O_vars = [pulp.LpVariable(f"O_{t+1}", lowBound=0) for t in range(T)]
        I_vars = [pulp.LpVariable(f"I_{t+1}", lowBound=0) for t in range(T)]
        B_vars = [pulp.LpVariable(f"B_{t+1}", lowBound=0) for t in range(T)]
        
        # Objective
        prob += pulp.lpSum([
            params['m'] * P_vars[t] +  # Material
            params['d'] * params['r'] * params['cr'] * W_vars[t] +  # Regular labor
            params['ch'] * H_vars[t] +  # Hiring
            params['cf'] * F_vars[t] +  # Firing
            params['co'] * O_vars[t] +  # Overtime
            params['h'] * I_vars[t] +  # Inventory
            params['b'] * B_vars[t]    # Backlog
            for t in range(T)
        ])
        
        # Constraints
        # Inventory balance
        for t in range(T):
            if t == 0:
                prob += I_vars[t] - B_vars[t] == params['I0'] - D[t] + P_vars[t]
            else:
                prob += I_vars[t] - B_vars[t] == I_vars[t-1] - B_vars[t-1] - D[t] + P_vars[t]
        
        # Workforce balance
        for t in range(T):
            if t == 0:
                prob += W_vars[t] == params['W0'] + H_vars[t] - F_vars[t]
            else:
                prob += W_vars[t] == W_vars[t-1] + H_vars[t] - F_vars[t]
        
        # Regular capacity
        for t in range(T):
            prob += P_vars[t] * params['s'] <= W_vars[t] * params['d'] * params['r'] + O_vars[t]
        
        # Overtime limit
        for t in range(T):
            prob += O_vars[t] <= W_vars[t] * params['OT_max']
        
        # Ending inventory
        prob += I_vars[T-1] >= params['I6']
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        
        if status == 1:  # Optimal
            P = np.array([pulp.value(p) for p in P_vars])
            W = np.array([pulp.value(w) for w in W_vars])
            H = np.array([pulp.value(h) for h in H_vars])
            F = np.array([pulp.value(f) for f in F_vars])
            O = np.array([pulp.value(o) for o in O_vars])
            I = np.array([pulp.value(i) for i in I_vars])
            B = np.array([pulp.value(b) for b in B_vars])
            results["Mixed (최적화)"] = {'P': P, 'W': W, 'H': H, 'F': F, 'O': O, 'I': I, 'B': B}
        else:
            st.error("LP 최적화 실패")
            results["Mixed (최적화)"] = None
    
    # Strategy 4: Overtime-Only
    if "Overtime-Only (초과근무)" in selected_strategies:
        W = np.full(T, params['W0'])
        capacity_regular = W * params['d'] * params['r'] / params['s']
        P = np.zeros(T)
        O = np.zeros(T)
        I = np.zeros(T+1)
        B = np.zeros(T+1)
        I[0] = params['I0']
        B[0] = 0
        
        for t in range(T):
            P[t] = capacity_regular[t]
            max_ot_production = W[t] * params['OT_max'] / params['s']
            P[t] += max_ot_production
            O[t] = W[t] * params['OT_max']
            
            I[t+1] = I[t] + P[t] - D[t]
            if I[t+1] < 0:
                B[t+1] = -I[t+1]
                I[t+1] = 0
        
        H = np.zeros(T)
        F = np.zeros(T)
        results["Overtime-Only (초과근무)"] = {'P': P, 'W': W, 'H': H, 'F': F, 'O': O, 'I': I[1:], 'B': B[1:]}
    
    return results

# Compute results
results = compute_strategies(st.session_state.params)

# Function to compute costs
def compute_costs(strategy_data, params):
    P, W, H, F, O, I, B = strategy_data['P'], strategy_data['W'], strategy_data['H'], strategy_data['F'], strategy_data['O'], strategy_data['I'], strategy_data['B']
    T = len(P)
    
    regular_labor = W * params['d'] * params['r'] * params['cr']
    overtime_cost = O * params['co']
    hiring_cost = H * params['ch']
    firing_cost = F * params['cf']
    inventory_cost = I * params['h']
    backlog_cost = B * params['b']
    material_cost = P * params['m']
    
    monthly_cost = regular_labor + overtime_cost + hiring_cost + firing_cost + inventory_cost + backlog_cost + material_cost
    total_cost = np.sum(monthly_cost)
    
    return {
        'monthly': monthly_cost,
        'total': total_cost,
        'breakdown': {
            'regular_labor': np.sum(regular_labor),
            'overtime': np.sum(overtime_cost),
            'hiring': np.sum(hiring_cost),
            'firing': np.sum(firing_cost),
            'inventory': np.sum(inventory_cost),
            'backlog': np.sum(backlog_cost),
            'material': np.sum(material_cost)
        }
    }

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 전략 비교 요약", "📅 월별 상세 계획", "📈 시각화 대시보드", "💡 계획 평가 및 권고"])

with tab1:
    st.header("📊 전략 비교 요약")
    
    summary_data = []
    for strategy, data in results.items():
        if data is None:
            continue
        costs = compute_costs(data, st.session_state.params)
        total_prod = np.sum(data['P'])
        avg_workforce = np.mean(data['W'])
        total_ot = np.sum(data['O'])
        total_inv = np.sum(data['I'])
        total_back = np.sum(data['B'])
        service_level = (len(data['B']) - np.count_nonzero(data['B'])) / len(data['B']) * 100
        
        summary_data.append({
            '전략': strategy,
            '총비용': costs['total'],
            '총생산': total_prod,
            '평균인력': avg_workforce,
            '총초과시간': total_ot,
            '총재고': total_inv,
            '총부재': total_back,
            '서비스레벨%': service_level
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.style.format({
        '총비용': '{:,.0f}',
        '총생산': '{:,.0f}',
        '평균인력': '{:.1f}',
        '총초과시간': '{:.1f}',
        '총재고': '{:,.0f}',
        '총부재': '{:,.0f}',
        '서비스레벨%': '{:.1f}%'
    }))
    
    # Bar chart
    fig = px.bar(df_summary, x='전략', y='총비용', title='전략별 총비용 비교', color='전략')
    st.plotly_chart(fig)
    
    # Highlight min cost
    min_cost_idx = df_summary['총비용'].idxmin()
    st.success(f"🏆 최저비용 전략: {df_summary.loc[min_cost_idx, '전략']} (총비용: {df_summary.loc[min_cost_idx, '총비용']:,.0f} 천원)")

with tab2:
    st.header("📅 월별 상세 계획")
    
    strategy_select = st.selectbox("전략 선택", list(results.keys()))
    if strategy_select in results and results[strategy_select] is not None:
        data = results[strategy_select]
        costs = compute_costs(data, st.session_state.params)
        
        months = ['1월', '2월', '3월', '4월', '5월', '6월']
        df_detail = pd.DataFrame({
            '월': months,
            '수요': st.session_state.params['demand'],
            '생산': data['P'],
            '인력': data['W'],
            '채용': data['H'],
            '해고': data['F'],
            '초과시간(h)': data['O'],
            '기말재고': data['I'],
            '부재': data['B'],
            '월비용': costs['monthly']
        })
        
        def color_cells(val):
            if val > 0:
                return 'background-color: red' if '부재' in str(val) else ''
            return ''
        
        st.dataframe(df_detail.style.format({
            '수요': '{:,.0f}',
            '생산': '{:,.0f}',
            '인력': '{:.0f}',
            '채용': '{:.0f}',
            '해고': '{:.0f}',
            '초과시간(h)': '{:.1f}',
            '기말재고': '{:,.0f}',
            '부재': '{:,.0f}',
            '월비용': '{:,.0f}'
        }).map(color_cells, subset=['부재']))

with tab3:
    st.header("📈 시각화 대시보드")
    
    col1, col2 = st.columns(2)
    
    # Chart 1: Demand vs Production
    with col1:
        fig1 = go.Figure()
        months = ['1월', '2월', '3월', '4월', '5월', '6월']
        fig1.add_trace(go.Bar(x=months, y=st.session_state.params['demand'], name='수요', marker_color='blue'))
        for strategy, data in results.items():
            if data:
                fig1.add_trace(go.Bar(x=months, y=data['P'], name=f'{strategy} 생산', opacity=0.7))
        fig1.update_layout(title='수요 vs 생산량', barmode='group')
        st.plotly_chart(fig1)
    
    # Chart 2: Inventory & Backlog
    with col2:
        fig2 = go.Figure()
        for strategy, data in results.items():
            if data:
                fig2.add_trace(go.Scatter(x=months, y=data['I'], mode='lines', name=f'{strategy} 재고', stackgroup='one'))
                fig2.add_trace(go.Scatter(x=months, y=data['B'], mode='lines', name=f'{strategy} 부재', stackgroup='two'))
        fig2.add_hline(y=st.session_state.params['I6'], line_dash="dot", annotation_text=f"목표 기말재고: {st.session_state.params['I6']}")
        fig2.update_layout(title='재고 및 부재 추이')
        st.plotly_chart(fig2)
    
    col3, col4 = st.columns(2)
    
    # Chart 3: Workforce Dynamics
    with col3:
        fig3 = go.Figure()
        for strategy, data in results.items():
            if data:
                fig3.add_trace(go.Scatter(x=months, y=data['W'], mode='lines+markers', name=f'{strategy} 인력'))
                fig3.add_trace(go.Bar(x=months, y=data['H'], name=f'{strategy} 채용', marker_color='green', opacity=0.5))
                fig3.add_trace(go.Bar(x=months, y=-data['F'], name=f'{strategy} 해고', marker_color='red', opacity=0.5))
        fig3.add_hline(y=st.session_state.params['W0'], line_dash="dot", annotation_text=f"초기 인력: {st.session_state.params['W0']}")
        fig3.update_layout(title='인력 변동', barmode='overlay')
        st.plotly_chart(fig3)
    
    # Chart 4: Cost Breakdown
    with col4:
        cost_data = []
        for strategy, data in results.items():
            if data:
                costs = compute_costs(data, st.session_state.params)
                for cost_type, amount in costs['breakdown'].items():
                    cost_data.append({'전략': strategy, '비용유형': cost_type, '금액': amount})
        df_costs = pd.DataFrame(cost_data)
        fig4 = px.bar(df_costs, x='전략', y='금액', color='비용유형', title='비용 구성', barmode='stack')
        st.plotly_chart(fig4)
    
    # Chart 5: Cumulative Cost
    fig5 = go.Figure()
    for strategy, data in results.items():
        if data:
            costs = compute_costs(data, st.session_state.params)
            cum_cost = np.cumsum(costs['monthly'])
            fig5.add_trace(go.Scatter(x=months, y=cum_cost, mode='lines+markers', name=strategy))
    fig5.update_layout(title='누적 비용 비교')
    st.plotly_chart(fig5)
    
    # Chart 6: Service Level
    service_levels = []
    for strategy, data in results.items():
        if data:
            sl = (len(data['B']) - np.count_nonzero(data['B'])) / len(data['B']) * 100
            service_levels.append({'전략': strategy, '서비스레벨%': sl})
    df_sl = pd.DataFrame(service_levels)
    fig6 = px.bar(df_sl, x='전략', y='서비스레벨%', title='서비스 레벨')
    st.plotly_chart(fig6)

with tab4:
    st.header("💡 계획 평가 및 권고")
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        min_cost_strategy = df_summary.loc[df_summary['총비용'].idxmin(), '전략']
        min_cost = df_summary['총비용'].min()
        max_cost = df_summary['총비용'].max()
        savings = max_cost - min_cost
        savings_pct = (savings / max_cost) * 100 if max_cost > 0 else 0
        
        st.write(f"**최저비용 전략:** {min_cost_strategy} (총비용: {min_cost:,.0f} 천원)")
        st.write(f"**비용 절감액:** {savings:,.0f} 천원 ({savings_pct:.1f}%)")
        
        # Check for backlog
        backlog_strategies = [s for s, d in results.items() if d and np.sum(d['B']) > 0]
        if backlog_strategies:
            st.warning(f"⚠️ 부재 발생 전략: {', '.join(backlog_strategies)} - 고객 서비스 리스크 있음")
        else:
            st.success("✅ 모든 전략에서 부재 없음")
        
        # Check overtime limits
        ot_limit_strategies = []
        for s, d in results.items():
            if d and np.any(d['O'] >= d['W'] * st.session_state.params['OT_max']):
                ot_limit_strategies.append(s)
        if ot_limit_strategies:
            st.warning(f"⚠️ 초과근무 한계 도달 전략: {', '.join(ot_limit_strategies)} - 운영 리스크 있음")
        
        # Workforce stability
        hiring_firing = {}
        for s, d in results.items():
            if d:
                total_h = np.sum(d['H'])
                total_f = np.sum(d['F'])
                hiring_firing[s] = total_h + total_f
        unstable = [s for s, v in hiring_firing.items() if v > 0]
        if unstable:
            st.info(f"ℹ️ 인력 변동 전략: {', '.join(unstable)} - 인력 안정성 고려 필요")
        
        # Recommendation
        st.subheader("📋 최종 권고")
        if min_cost_strategy == "Mixed (최적화)":
            st.success("혼합 최적화 전략을 권장합니다. 비용 효율성과 유연성을 모두 고려한 최적 솔루션입니다.")
        elif min_cost_strategy == "Level Production (평준화)":
            st.info("평준화 전략을 권장합니다. 인력 안정성과 예측 가능성이 높습니다.")
        elif min_cost_strategy == "Chase Demand (추종)":
            st.info("추종 전략을 고려해보세요. 수요 변동에 민첩하게 대응할 수 있습니다.")
        else:
            st.info("초과근무 전략은 단기간 운영에 적합할 수 있지만, 장기적으로는 리스크가 있습니다.")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("최저비용 전략", min_cost_strategy)
    with col2:
        st.metric("총비용 절감액", f"{savings:,.0f} 천원")
    with col3:
        max_inv = max([np.max(d['I']) for d in results.values() if d] + [0])
        st.metric("최대재고", f"{max_inv:,.0f} 단위")
    with col4:
        total_back = sum([np.sum(d['B']) for d in results.values() if d])
        st.metric("총부재고", f"{total_back:,.0f} 단위")

# Formulas expander
with st.expander("📐 수식 및 계산 로직 보기"):
    st.markdown("""
    ### 주요 수식
    
    **재고 균형:**
    ```
    I_t - B_t = I_{t-1} - B_{t-1} + P_t - D_t
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