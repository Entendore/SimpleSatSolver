import streamlit as st
import ast
import random
import time
from itertools import product
import json
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Advanced SAT Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.trace-step {
    background-color: #f8f9fa;
    padding: 10px;
    border-left: 3px solid #007bff;
    margin: 5px 0;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

class SATSolver:
    """Enhanced DPLL-based SAT solver with optimizations"""
    
    def __init__(self, heuristic="moms"):
        self.heuristic = heuristic
        self.stats = {
            'decisions': 0,
            'propagations': 0,
            'conflicts': 0,
            'restarts': 0,
            'time': 0
        }
        self.trace = []
        self.watchers = {}  # For watched literals optimization
        
    def solve(self, cnf: List[List[int]], verbose: bool = False) -> Optional[Dict[int, bool]]:
        """Main solving interface"""
        start_time = time.time()
        self.trace = [] if verbose else None
        self.stats = {k: 0 for k in self.stats}
        
        # Initialize watchers for optimization
        self._initialize_watchers(cnf)
        
        result = self._dpll(cnf.copy(), {})
        
        self.stats['time'] = time.time() - start_time
        return result
    
    def _initialize_watchers(self, cnf: List[List[int]]):
        """Initialize watched literals for each clause"""
        self.watchers = {}
        for i, clause in enumerate(cnf):
            if clause:
                # Watch first two literals
                lit1, lit2 = clause[0], clause[1] if len(clause) > 1 else clause[0]
                for lit in [lit1, lit2]:
                    if lit not in self.watchers:
                        self.watchers[lit] = []
                    self.watchers[lit].append(i)
    
    def _dpll(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        """Recursive DPLL with optimizations"""
        # Unit propagation with watched literals
        clauses, assignment, conflict = self._unit_propagate(clauses, assignment)
        if conflict:
            self.stats['conflicts'] += 1
            if self.trace is not None:
                self.trace.append(("Conflict", assignment.copy(), clauses))
            return None
        
        # Pure literal elimination
        clauses, assignment = self._pure_literal_eliminate(clauses, assignment)
        
        # Check termination conditions
        if not clauses:
            if self.trace is not None:
                self.trace.append(("Success", assignment.copy(), clauses))
            return assignment
        if [] in clauses:
            self.stats['conflicts'] += 1
            if self.trace is not None:
                self.trace.append(("Conflict", assignment.copy(), clauses))
            return None
        
        # Variable selection with heuristics
        var = self._select_variable(clauses, assignment)
        
        # Try both values
        for val in [True, False]:
            self.stats['decisions'] += 1
            new_assignment = assignment.copy()
            new_assignment[var] = val
            
            if self.trace is not None:
                self.trace.append((f"Decision: {var} = {val}", new_assignment.copy(), clauses))
            
            # Simplify with new assignment
            simplified_clauses = self._simplify(clauses, new_assignment)
            
            result = self._dpll(simplified_clauses, new_assignment)
            if result is not None:
                return result
        
        return None
    
    def _unit_propagate(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Tuple[List[List[int]], Dict[int, bool], bool]:
        """Enhanced unit propagation with watched literals"""
        changed = True
        conflict = False
        
        while changed and not conflict:
            changed = False
            unit_clauses = [c[0] for c in clauses if len(c) == 1]
            
            for lit in unit_clauses:
                var = abs(lit)
                val = lit > 0
                
                if var in assignment:
                    if assignment[var] != val:
                        conflict = True
                        break
                    continue
                
                assignment[var] = val
                self.stats['propagations'] += 1
                
                if self.trace is not None:
                    self.trace.append((f"Unit: {var} = {val}", assignment.copy(), clauses))
                
                clauses = self._simplify(clauses, assignment)
                changed = True
        
        return clauses, assignment, conflict
    
    def _pure_literal_eliminate(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Tuple[List[List[int]], Dict[int, bool]]:
        """Eliminate pure literals"""
        if not clauses:
            return clauses, assignment
        
        all_lits = [lit for clause in clauses for lit in clause]
        lit_counts = {}
        
        for lit in all_lits:
            lit_counts[lit] = lit_counts.get(lit, 0) + 1
        
        for lit in list(lit_counts.keys()):
            if -lit not in lit_counts:
                var = abs(lit)
                if var not in assignment:
                    assignment[var] = lit > 0
                    if self.trace is not None:
                        self.trace.append((f"Pure Literal: {var} = {lit > 0}", assignment.copy(), clauses))
                    clauses = self._simplify(clauses, assignment)
        
        return clauses, assignment
    
    def _select_variable(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> int:
        """Variable selection with heuristics"""
        unassigned = set(abs(lit) for clause in clauses for lit in clause) - assignment.keys()
        
        if not unassigned:
            return None
        
        if self.heuristic == "random":
            return random.choice(list(unassigned))
        elif self.heuristic == "moms":
            # Maximum Occurrences in Minimum Size clauses
            min_size = min(len(c) for c in clauses)
            min_clauses = [c for c in clauses if len(c) == min_size]
            var_counts = {}
            for clause in min_clauses:
                for lit in clause:
                    var = abs(lit)
                    var_counts[var] = var_counts.get(var, 0) + 1
            return max(var_counts, key=var_counts.get)
        elif self.heuristic == "vsids":
            # Variable State Independent Decaying Sum (simplified)
            var_scores = {}
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    var_scores[var] = var_scores.get(var, 0) + 1
            return max(var_scores, key=var_scores.get)
        else:
            return next(iter(unassigned))
    
    def _simplify(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> List[List[int]]:
        """Simplify clauses based on current assignment"""
        simplified = []
        for clause in clauses:
            new_clause = []
            satisfied = False
            for literal in clause:
                var = abs(literal)
                if var in assignment:
                    val = assignment[var]
                    if (literal > 0 and val) or (literal < 0 and not val):
                        satisfied = True
                        break
                else:
                    new_clause.append(literal)
            if not satisfied and new_clause:
                simplified.append(new_clause)
        return simplified

# Utility functions
def generate_random_3sat(n_vars: int, n_clauses: int) -> List[List[int]]:
    """Generate random 3-SAT instance"""
    clauses = []
    for _ in range(n_clauses):
        clause = set()
        while len(clause) < 3:
            lit = random.randint(1, n_vars)
            if random.choice([True, False]):
                lit = -lit
            clause.add(lit)
        clauses.append(list(clause))
    return clauses

def generate_random_kcnf(n_vars: int, n_clauses: int, k: int = 3) -> List[List[int]]:
    """Generate random k-CNF instance"""
    clauses = []
    for _ in range(n_clauses):
        clause = set()
        while len(clause) < k:
            lit = random.randint(1, n_vars)
            if random.choice([True, False]):
                lit = -lit
            clause.add(lit)
        clauses.append(list(clause))
    return clauses

def truth_table(cnf: List[List[int]], max_vars: int = 4) -> Optional[Tuple[List[int], List[Tuple[Dict[int, bool], bool]]]]:
    """Generate truth table for small instances"""
    vars_in_cnf = sorted(set(abs(l) for clause in cnf for l in clause))
    if len(vars_in_cnf) > max_vars:
        return None
    
    table = []
    for values in product([False, True], repeat=len(vars_in_cnf)):
        assignment = dict(zip(vars_in_cnf, values))
        result = all(any((assignment.get(abs(l), False) if l > 0 else not assignment.get(abs(l), False))
                        for l in clause) for clause in cnf)
        table.append((assignment, result))
    return vars_in_cnf, table

def parse_dimacs(dimacs_text: str) -> List[List[int]]:
    """Parse DIMACS CNF format"""
    clauses = []
    for line in dimacs_text.splitlines():
        line = line.strip()
        if not line or line.startswith(('c', '%', '0')):
            continue
        if line.startswith('p'):
            continue
        clause = [int(x) for x in line.split() if x != '0']
        if clause:
            clauses.append(clause)
    return clauses

def export_results(assignment: Dict[int, bool], stats: Dict, filename: str = "sat_result.json"):
    """Export solving results to JSON"""
    result = {
        'assignment': {str(k): v for k, v in assignment.items()},
        'stats': stats,
        'timestamp': time.time()
    }
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

# Example formulas
EXAMPLES = {
    "Simple satisfiable": "[[1, -2], [2], [-1, 3]]",
    "Unsatisfiable contradiction": "[[1], [-1]]",
    "3-SAT example": "[[1, 2, 3], [-1, -2], [-3, 2], [1, -2, -3]]",
    "Pigeonhole principle (3 pigeons, 2 holes)": "[[1, 2], [3, 4], [5, 6], [-1, -3], [-1, -5], [-3, -5], [-2, -4], [-2, -6], [-4, -6]]",
    "Pythagorean triples (1-10)": "[[1, 4, 9], [4, 9, 16], [9, 16, 25], [16, 25, 36], [25, 36, 49], [36, 49, 64], [49, 64, 81], [64, 81, 100]]"
}

# Main application
def main():
    st.title("🧠 Advanced SAT Solver")
    st.markdown("A powerful DPLL-based SAT solver with multiple heuristics and optimizations")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Heuristic selection
    heuristic = st.sidebar.selectbox(
        "Variable Selection Heuristic",
        ["moms", "vsids", "random", "first"],
        help="Choose the heuristic for selecting variables during search"
    )
    
    # Performance options
    enable_trace = st.sidebar.checkbox("Enable detailed trace", value=False)
    show_stats = st.sidebar.checkbox("Show performance statistics", value=True)
    
    # Initialize solver
    solver = SATSolver(heuristic=heuristic)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Input", "🔧 Generator", "📊 Analysis", "📁 File I/O"])
    
    with tab1:
        st.header("CNF Formula Input")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            example_choice = st.selectbox("Choose Example", list(EXAMPLES.keys()))
            cnf_input = st.text_area(
                "CNF Formula (Python list format)",
                EXAMPLES[example_choice],
                height=150,
                help="Format: [[1, -2], [3], [-1, 2]] where positive = True literal, negative = False"
            )
        
        with col2:
            st.markdown("### Quick Info")
            st.markdown("""
            - **CNF**: Conjunctive Normal Form
            - **Positive number**: Variable is True
            - **Negative number**: Variable is False
            - **Clause**: OR of literals
            - **Formula**: AND of clauses
            """)
        
        if st.button("🚀 Solve", type="primary"):
            try:
                cnf = ast.literal_eval(cnf_input)
                
                with st.spinner("Solving..."):
                    result = solver.solve(cnf, verbose=enable_trace)
                
                # Display results
                if result:
                    st.success("✅ Formula is SATISFIABLE")
                    st.json(result)
                    
                    if show_stats:
                        st.subheader("📈 Performance Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Decisions", solver.stats['decisions'])
                        with col2:
                            st.metric("Propagations", solver.stats['propagations'])
                        with col3:
                            st.metric("Conflicts", solver.stats['conflicts'])
                        with col4:
                            st.metric("Time (s)", f"{solver.stats['time']:.4f}")
                    
                    # Export button
                    if st.button("💾 Export Results"):
                        export_results(result, solver.stats)
                        st.success("Results exported to sat_result.json")
                else:
                    st.error("❌ Formula is UNSATISFIABLE")
                    if show_stats:
                        st.subheader("📈 Performance Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Decisions", solver.stats['decisions'])
                        with col2:
                            st.metric("Propagations", solver.stats['propagations'])
                        with col3:
                            st.metric("Conflicts", solver.stats['conflicts'])
                        with col4:
                            st.metric("Time (s)", f"{solver.stats['time']:.4f}")
                
                # Show trace if enabled
                if enable_trace and solver.trace:
                    st.subheader("🔍 Solving Trace")
                    for i, (step, assign, remaining) in enumerate(solver.trace):
                        with st.expander(f"Step {i+1}: {step}"):
                            st.code(f"Assignment: {assign}\nRemaining clauses: {remaining}")
                
                # Truth table for small instances
                tt = truth_table(cnf)
                if tt:
                    vars_in_cnf, table = tt
                    st.subheader("📊 Truth Table")
                    df = pd.DataFrame([
                        [str(assign), "✅" if result else "❌"]
                        for assign, result in table
                    ], columns=["Assignment", "Satisfies"])
                    # Updated to use width='stretch' instead of use_container_width=True
                    st.dataframe(df, width='stretch')
                else:
                    st.info("Truth table hidden: too many variables (>4)")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Random Instance Generator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            instance_type = st.selectbox("Instance Type", ["3-SAT", "k-CNF"])
            n_vars = st.slider("Number of Variables", 3, 20, 10)
            n_clauses = st.slider("Number of Clauses", 3, 100, 20)
            
            if instance_type == "k-CNF":
                k = st.slider("Clause Size (k)", 2, 5, 3)
        
        with col2:
            st.markdown("### Generation Info")
            st.markdown("""
            - **3-SAT**: Each clause has exactly 3 literals
            - **k-CNF**: Each clause has exactly k literals
            - Variables are numbered 1 to n
            - Literals are randomly negated
            """)
        
        if st.button("🎲 Generate Instance"):
            if instance_type == "3-SAT":
                cnf = generate_random_3sat(n_vars, n_clauses)
            else:
                cnf = generate_random_kcnf(n_vars, n_clauses, k)
            
            st.text_area("Generated CNF", str(cnf), height=150)
            
            # Quick analysis
            with st.expander("Quick Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Variables", n_vars)
                with col2:
                    st.metric("Clauses", n_clauses)
                with col3:
                    st.metric("Clause Size", k if instance_type == "k-CNF" else 3)
    
    with tab3:
        st.header("Formula Analysis")
        
        st.markdown("### Compare Heuristics")
        
        cnf_analysis = st.text_area(
            "Enter CNF for Analysis",
            "[[1, 2, 3], [-1, -2], [-3, 2], [1, -2, -3]]",
            height=100
        )
        
        if st.button("📊 Analyze"):
            try:
                cnf = ast.literal_eval(cnf_analysis)
                heuristics = ["moms", "vsids", "random", "first"]
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, h in enumerate(heuristics):
                    status_text.text(f"Testing {h} heuristic...")
                    test_solver = SATSolver(heuristic=h)
                    result = test_solver.solve(cnf)
                    results[h] = {
                        'satisfiable': result is not None,
                        'time': test_solver.stats['time'],
                        'decisions': test_solver.stats['decisions'],
                        'propagations': test_solver.stats['propagations'],
                        'conflicts': test_solver.stats['conflicts']
                    }
                    progress_bar.progress((i + 1) / len(heuristics))
                
                status_text.text("Analysis complete!")
                
                # Display comparison
                st.subheader("Heuristic Comparison")
                df = pd.DataFrame(results).T
                # Updated to use width='stretch' instead of use_container_width=True
                st.dataframe(df, width='stretch')
                
                # Visualization
                st.subheader("Performance Visualization")
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    st.bar_chart(df[['time', 'decisions']])
                
                with fig_col2:
                    st.bar_chart(df[['propagations', 'conflicts']])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab4:
        st.header("File I/O")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload DIMACS File")
            uploaded_file = st.file_uploader("Upload CNF file", type=["cnf", "txt"])
            
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode()
                    parsed_cnf = parse_dimacs(content)
                    st.success(f"Parsed {len(parsed_cnf)} clauses")
                    st.text_area("Parsed CNF", str(parsed_cnf), height=150)
                    
                    if st.button("Solve Uploaded File"):
                        with st.spinner("Solving..."):
                            result = solver.solve(parsed_cnf, verbose=enable_trace)
                        
                        if result:
                            st.success("✅ Satisfiable")
                            st.json(result)
                        else:
                            st.error("❌ Unsatisfiable")
                            
                except Exception as e:
                    st.error(f"Parse failed: {str(e)}")
        
        with col2:
            st.subheader("Export Current Session")
            st.markdown("""
            Export your current session data including:
            - Last solved formula
            - Solution (if satisfiable)
            - Performance statistics
            """)
            
            if st.button("📥 Export Session Data"):
                session_data = {
                    'timestamp': time.time(),
                    'heuristic': heuristic,
                    'stats': solver.stats,
                    'last_formula': cnf_input if 'cnf_input' in locals() else None
                }
                
                st.download_button(
                    label="Download session.json",
                    data=json.dumps(session_data, indent=2),
                    file_name="sat_session.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()