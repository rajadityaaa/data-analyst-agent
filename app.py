import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import agent

MAX_FILE_SIZE_MB = 10
MAX_ROWS = 50000

st.set_page_config(
    page_title="CSV Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

# Custom minimal CSS styling
st.markdown("""
<style>
  .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  .main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2B6CB0;
    margin-bottom: 0.1rem;
  }
  .sub-title {
    font-size: 1.05rem;
    color: #718096;
    margin-bottom: 1.5rem;
  }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">CSV Data Analyst Agent 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a CSV dataset and ask natural-language questions to receive tool-verified calculations.</div>', unsafe_allow_html=True)

# Top Bar CSV File Uploader
with st.container(border=True):
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"], help="Upload CSV files up to 10MB (max 50,000 rows)")

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB. Please upload a smaller file.")
        st.session_state.pop("df", None)
    else:
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

            if df.empty or len(df) == 0:
                st.error("The uploaded CSV file is empty (0 rows). Please upload a dataset with data.")
                st.session_state.pop("df", None)
            elif len(df) > MAX_ROWS:
                st.error(f"Dataset has {len(df):,} rows, which exceeds the limit of {MAX_ROWS:,} rows. Please upload a smaller dataset.")
                st.session_state.pop("df", None)
            else:
                if st.session_state.get("filename") != uploaded_file.name:
                    st.session_state["df"] = df
                    st.session_state["filename"] = uploaded_file.name
                    st.toast(f"Loaded **{uploaded_file.name}** successfully! 🎉")

        except Exception as e:
            st.error(f"Failed to parse CSV file: {str(e)}")
            st.session_state.pop("df", None)
else:
    st.session_state.pop("df", None)


def render_visualization(tool_name: str, arguments: dict, evidence: dict, preferred_style: str = "Bar Chart"):
    """Renders a flexible rule-based Matplotlib chart based on evidence and user toggle."""
    if not isinstance(evidence, dict) or "error" in evidence:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    chart_rendered = False

    if tool_name == "group_and_aggregate" and "results" in evidence:
        results = evidence.get("results", [])
        if len(results) > 1:
            groups = [str(r.get("group", "N/A")) for r in results[:10]]
            values = [r.get("value", 0) for r in results[:10]]
            group_col = str(evidence.get("group_by", "")).lower()

            if preferred_style == "Line Chart" or (preferred_style == "Auto" and any(t in group_col for t in ["date", "year", "month", "time", "day"])):
                ax.plot(groups, values, marker="o", color="#2563EB", linewidth=2.5)
                ax.grid(True, linestyle="--", alpha=0.5)
            elif preferred_style == "Horizontal Bar":
                groups.reverse()
                values.reverse()
                ax.barh(groups, values, color="#4F46E5", alpha=0.85)
            else:
                ax.bar(groups, values, color="#4F46E5", alpha=0.85)
                labels = [g[:15] + "..." if len(g) > 15 else g for g in groups]
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels(labels, rotation=25 if len(max(labels, key=len)) > 8 else 0, ha="right")

            ax.set_title(f"{str(evidence.get('operation', '')).capitalize()} of {evidence.get('metric', '')} by {evidence.get('group_by', '')}", fontsize=11, fontweight="bold")
            ax.set_ylabel(str(evidence.get("metric", "")))
            ax.set_xlabel(str(evidence.get("group_by", "")))
            plt.tight_layout()
            chart_rendered = True

    elif tool_name == "top_values" and "results" in evidence:
        results = evidence.get("results", [])
        sort_by = evidence.get("sorted_by") or evidence.get("target_column")
        target_col = evidence.get("target_column")

        if len(results) > 1 and sort_by and target_col:
            names = []
            values = []
            for r in results[:10]:
                val = r.get(sort_by)
                if isinstance(val, (int, float)):
                    name_str = str(r.get(target_col, "N/A"))
                    names.append(name_str[:20] + "..." if len(name_str) > 20 else name_str)
                    values.append(val)

            if len(values) > 1:
                if preferred_style == "Bar Chart":
                    ax.bar(names, values, color="#0EA5E9", alpha=0.85)
                    ax.set_xticklabels(names, rotation=25, ha="right")
                elif preferred_style == "Line Chart":
                    ax.plot(names, values, marker="s", color="#0EA5E9", linewidth=2.5)
                    ax.grid(True, linestyle="--", alpha=0.5)
                else:  # Horizontal Bar default for top values
                    names.reverse()
                    values.reverse()
                    ax.barh(names, values, color="#0EA5E9", alpha=0.85)

                ax.set_title(f"Top {len(values)} Items by {sort_by}", fontsize=11, fontweight="bold")
                ax.set_xlabel(str(sort_by))
                plt.tight_layout()
                chart_rendered = True

    if chart_rendered:
        st.pyplot(fig)
    else:
        plt.close(fig)


# Display Tabbed UI if DataFrame is loaded
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]

    tab_qa, tab_overview = st.tabs(["🤖 Analyst Agent & Q&A", "📋 Dataset Overview"])

    # ---------------- TAB 1: ANALYST AGENT & Q&A ----------------
    with tab_qa:
        with st.container(border=True):
            st.markdown("### 💡 Example Questions (click to populate)")
            chip_cols = st.columns(4)
            example_questions = [
                "Which region generated the most revenue?",
                "What are the top 5 products by revenue?",
                "What is the average profit across all orders?",
                "Which product category has the highest sales?"
            ]

            def populate_question(q_text):
                st.session_state["qa_input"] = q_text

            for idx, q_text in enumerate(example_questions):
                chip_cols[idx].button(
                    q_text,
                    key=f"chip_{idx}",
                    on_click=populate_question,
                    args=(q_text,)
                )

            question_input = st.text_input(
                "Ask a natural-language question about your dataset:",
                placeholder="e.g. Which region generated the most revenue?",
                key="qa_input"
            )

            analyze_btn = st.button("🚀 Analyze Question", type="primary")

            if analyze_btn:
                question_to_run = st.session_state.get("qa_input", "").strip()
                if not question_to_run:
                    st.warning("Please enter a valid question before analyzing.")
                else:
                    with st.status("Analyzing your question...", expanded=True) as status:
                        status.write("🧠 **Step 1:** Interpreting question intent & selecting Pandas tool...")
                        time.sleep(0.3)
                        status.write("⚡ **Step 2:** Executing exact Pandas computation...")
                        try:
                            result = agent.run_agent(question_to_run, df)
                            status.write("✍️ **Step 3:** Synthesizing verified natural language explanation...")
                            time.sleep(0.2)
                            st.session_state["last_result"] = result

                            if result.get("status") == "success":
                                status.update(label="Analysis complete! ✅", state="complete", expanded=False)
                                st.toast("Analysis complete! ✅")
                            else:
                                status.update(label="Analysis finished with message ⚠️", state="error", expanded=False)
                                st.toast("Analysis notice ⚠️")

                        except Exception as e:
                            status.update(label="Analysis failed ❌", state="error", expanded=False)
                            st.error(f"An unexpected error occurred during analysis: {str(e)}")
                            st.session_state.pop("last_result", None)

        # Render Agent Results
        if "last_result" in st.session_state and st.session_state["last_result"] is not None:
            res = st.session_state["last_result"]

            if res.get("status") == "error":
                st.error(res.get("message", "Could not complete analysis."))
            elif res.get("status") == "success":
                st.markdown("---")
                
                # Answer Card
                with st.container(border=True):
                    st.markdown("### 💡 Natural Language Answer")
                    st.caption(f"🔧 **Tool Selected:** `{res.get('tool_selected')}` | **Arguments:** `{res.get('arguments')}`")
                    st.markdown(res.get("explanation", ""))

                evidence = res.get("evidence")
                has_chart = res.get("tool_selected") in ["group_and_aggregate", "top_values"] and isinstance(evidence, dict) and "results" in evidence and len(evidence.get("results", [])) > 1

                # Results Layout (Columns if chart exists)
                if has_chart:
                    col_ev, col_chart = st.columns([1, 1])
                    
                    with col_ev:
                        with st.container(border=True):
                            st.subheader("🔢 Verified Evidence Table")
                            evidence_df = pd.DataFrame(evidence["results"])
                            st.dataframe(evidence_df, use_container_width=True)
                            
                            # CSV Download Button
                            csv_data = evidence_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Evidence CSV",
                                data=csv_data,
                                file_name=f"evidence_{res.get('tool_selected')}.csv",
                                mime="text/csv",
                                type="secondary"
                            )

                    with col_chart:
                        with st.container(border=True):
                            st.subheader("📊 Data Visualization")
                            chart_style = st.radio(
                                "Select Chart Style:",
                                options=["Bar Chart", "Horizontal Bar", "Line Chart"],
                                horizontal=True,
                                key="chart_style_radio"
                            )
                            render_visualization(res.get("tool_selected"), res.get("arguments", {}), evidence, preferred_style=chart_style)

                else:
                    with st.container(border=True):
                        st.subheader("🔢 Verified Evidence")
                        if isinstance(evidence, dict) and "results" in evidence and isinstance(evidence["results"], list):
                            evidence_df = pd.DataFrame(evidence["results"])
                            st.dataframe(evidence_df, use_container_width=True)
                            csv_data = evidence_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Evidence CSV",
                                data=csv_data,
                                file_name=f"evidence_{res.get('tool_selected')}.csv",
                                mime="text/csv"
                            )
                        elif isinstance(evidence, dict) and "preview" in evidence and isinstance(evidence["preview"], list):
                            evidence_df = pd.DataFrame(evidence["preview"])
                            st.dataframe(evidence_df, use_container_width=True)
                        else:
                            st.json(evidence)

                # Raw Output Expander
                with st.expander("🔍 Raw Tool Computation JSON"):
                    st.json(evidence)

    # ---------------- TAB 2: DATASET OVERVIEW ----------------
    with tab_overview:
        with st.container(border=True):
            st.header("📋 Dataset Dashboard")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Rows", f"{len(df):,}")
            m2.metric("Total Columns", f"{len(df.columns):,}")
            m3.metric("Total Missing Values", f"{df.isnull().sum().sum():,}")

        # Column Schema Info
        with st.container(border=True):
            st.subheader("Column Schema & Data Types")
            col_info = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": [str(dtype) for dtype in df.dtypes],
                "Missing Values": df.isnull().sum().values,
                "Missing Percentage": [f"{(count / len(df)) * 100:.1f}%" for count in df.isnull().sum().values]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)

        # Full Data Preview Expander
        with st.expander("🔍 Full Data Preview (First 50 Rows)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
