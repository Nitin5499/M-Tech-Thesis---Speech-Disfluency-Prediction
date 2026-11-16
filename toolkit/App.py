"""
app.py -- small web page for the toolkit (Streamlit).

run from the project folder:
  streamlit run toolkit/app.py
"""

import streamlit as st
import pandas as pd
from disfluency_toolkit import Toolkit, TASKS
from restart_rule import find_restarts

st.set_page_config(page_title="Disfluency predictor", layout="wide")


@st.cache_resource(show_spinner="Loading the models (the first start takes a minute)...")
def get_toolkit():
    return Toolkit(verbose=False)


tk = get_toolkit()

st.title("Predicting speech disfluency from typed text")
st.caption("Type a sentence. For every gap between two words the toolkit gives the chance "
           "of a filler (uh / um) and of a reparandum (a word that gets repaired).")

text = st.text_area("Sentence(s)", "so i went to the store and then we got the car and drove home", height=90)
use_rule = st.checkbox("Mark repairs with the repetition rule (a phrase that comes back after a short break)", value=True)
c1, c2 = st.columns(2)
thr_f = c1.slider("Mark a filler when the chance is at least", 0.30, 0.90, 0.50, 0.05)
thr_r = c2.slider("Mark a reparandum when the chance is at least", 0.30, 0.90, 0.70, 0.05)

if st.button("Predict", type="primary"):
    with st.spinner("computing features..."):
        res = tk.analyze(text)
    if res.empty:
        st.warning("Please type at least two words.")
    else:
        st.subheader("Where a disfluency is likely")
        st.caption("red = filler (uh / um) likely at this gap (model);  blue struck-through words = reparandum, "
                   "green = the repair. With the box ticked, repairs come from the repetition rule only "
                   "(a phrase that comes back); untick it to strike words by the model chance and the slider.")
        for s_no, g in res.groupby("sentence"):
            words = [g.iloc[0]["word_before"]] + g["word_after"].tolist()
            n = len(g)
            pf = g["P_filler"].tolist()
            pr = g["P_reparandum"].tolist()
            marked = []
            restarts = find_restarts(words) if use_rule else []
            role = {}
            for r in restarts:
                for k in range(*r["reparandum"]):
                    role[k] = "rep"
                for k in r["edit"]:
                    role[k] = "edit"
                for k in range(*r["repair"]):
                    role[k] = "fix"
            for i, w in enumerate(words):
                if use_rule:
                    if role.get(i) == "rep":
                        marked.append(f":blue[~~{w}~~]")
                    elif role.get(i) == "edit":
                        marked.append(f":gray[*{w}*]")
                    elif role.get(i) == "fix":
                        if role.get(i - 1) != "fix":
                            marked.append(":green[→ repair:]")
                        marked.append(f":green[**{w}**]")
                    else:
                        marked.append(f"**{w}**")
                else:
                    # gap i sits right after word i; a reparandum is judged on word i
                    if i < n and pr[i] >= thr_r:
                        marked.append(f":blue[~~{w}~~]")
                        marked.append(f":blue[→ repair: **{words[i + 1]}** ({pr[i]:.2f})]")
                    else:
                        marked.append(f"**{w}**")
                # no filler mark between two words of the same struck or repaired phrase
                inside = use_rule and role.get(i) in ("rep", "fix") and role.get(i) == role.get(i + 1)
                if i < n and pf[i] >= thr_f and not inside:
                    marked.append(f":red[‹filler {pf[i]:.2f}›]")
            st.markdown(" ".join(marked))

        st.subheader("Gap by gap")
        show = res[["sentence", "gap", "word_before", "word_after", "P_filler", "P_reparandum"]]
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "P_filler": st.column_config.ProgressColumn("P(filler)", min_value=0.0, max_value=1.0, format="%.2f"),
                "P_reparandum": st.column_config.ProgressColumn("P(reparandum)", min_value=0.0, max_value=1.0, format="%.2f"),
            })

        with st.expander("Features behind the numbers"):
            st.dataframe(res.drop(columns=["P_filler", "P_reparandum"]).round(3), hide_index=True)

st.divider()
for t in TASKS:
    r = tk.meta.loc[t]
    st.caption(f"{t}: held-out accuracy of this text-only model {tk.accuracy(t):.2f}% "
               f"(always guessing the majority class gives {r.always_guess:.2f}%, {int(r.n_rows):,} Switchboard rows). "
               "Accuracy is measured at a 0.50 cutoff. Probabilities are relative to that training sample, so read them as a ranking of gaps.")
st.caption("The accuracy numbers above belong to the statistical model. Repair marking by the repetition rule "
           "is a separate step that only looks at the typed words.")