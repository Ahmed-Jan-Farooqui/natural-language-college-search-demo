import streamlit as st
import asyncio
import json
import time
import numpy as np
from pydantic import BaseModel
from agents import Agent, Runner, trace, WebSearchTool
from openai import OpenAI

# -----------------------------
# Utilities
# -----------------------------

def cosine_similarity(vec_a, vec_b):
    A = np.array(vec_a)
    B = np.array(vec_b)
    dot = np.dot(A, B)
    na = np.linalg.norm(A)
    nb = np.linalg.norm(B)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# -----------------------------
# Models
# -----------------------------

class SearchResult(BaseModel):
    college_names: list[str]
    references: list[str]


class TranslationResult(BaseModel):
    college_names: list[str]

class QueryStructure(BaseModel):
    subqueries: list[str]   # ["party schools", "California", "skiing nearby"]
    logic: str              # "(Q1 AND Q2) OR Q3"



# -----------------------------
# Agents
# -----------------------------

query_splitter_agent_prompt = """
You are an expert query-decomposition agent.

### Your task:
Given a user natural-language query (e.g. "great CS schools in California or New York with good snowboarding"),
produce:

1. A list of **atomic sub-queries**, each self-contained and suitable for web search.
2. A **Boolean logic expression** describing how they relate.

### Rules:
- Sub-queries should be the minimum units that can be searched independently.
- Use parentheses in the logic expression.
- Use only `AND`, `OR`, `NOT`.
- The logic must reference sub-queries as Q1, Q2, Q3... in order.

### Example:
User: "good engineering colleges near mountains AND skiing OR big party schools"
Output:
{
  "subqueries": [
    "engineering colleges near mountains",
    "colleges near skiing locations",
    "big party schools"
  ],
  "logic": "(Q1 AND Q2) OR Q3"
}

Return output in the QueryStructure schema.
"""

search_agent_prompt = '''
You are a multi-step research agent that receives a structured decomposition
of a user query, consisting of:
- A list of atomic sub-queries: Q1, Q2, Q3...
- A boolean logic expression describing how to combine results: e.g. "(Q1 AND Q2) OR Q3"

Your input will be of the form:
{
  "subqueries": [...],
  "logic": "(Q1 AND Q2) OR Q3"
}

### Your responsibilities:

1. **Write a short execution plan** before searching.
2. For EACH subquery Qi:
   - Transform Qi into 1–2 well-targeted web search queries.
   - Call the web search tool for each search query.
3. For each Qi:
   - Extract the set of colleges found.
   - Maintain a mapping: Qi → {set of colleges}
   - **Include colleges even if the confidence that they match the subquery is moderate (~70%)**
   - If evidence is weak but the college seems related, still include it.
4. Combine the sets using the boolean logic expression:
   - AND = set intersection
   - OR  = set union
   - NOT = set complement (relative to all schools in any Qi)
5. Rank final schools based on:
   - frequency of appearance
   - strength of evidence from search results
   - **Include borderline matches lower in the ranking, but include as many as reasonably possible.**
6. Return:
   - final list of college names (include all borderline matches)
   - list of references (all tool outputs)

### Important Rules:
- You MUST use the web tool at least once per subquery.
- You MUST follow the logic expression literally.
- Do NOT exclude colleges just because they are not a perfect match; include lower-confidence matches.

Return the final output according to the SearchResult schema.
'''



search_agent = Agent(
    name="Search Agent",
    instructions=search_agent_prompt,
    tools=[WebSearchTool()],
    output_type=SearchResult
)

query_splitter_agent = Agent(
    name="Query Splitter Agent",
    instructions=query_splitter_agent_prompt,
    output_type=QueryStructure
)

translation_agent = Agent(
    name="Translation Agent",
    instructions='''
    You are a helpful agent that will be given input of the following form:
        { <college_name> : [list of candidate names], ... }
    Your job is to figure out which candidate name corresponds to the actual name,
    and return that.
    ''',
    output_type=TranslationResult
)

client = OpenAI()
MODEL_NAME = "text-embedding-3-small"
BATCH_SIZE = 1000


# -----------------------------
# MAIN WORKFLOW (converted for Streamlit)
# -----------------------------

async def pipeline(query: str, trace_box):

    # Load embeddings + metadata
    with open("data/embeddings.json", "r") as f:
        embeddings_dict = dict(json.load(f))

    with open("data/current-colleges.json", "r") as f:
        temp = json.load(f)
        colleges = {c["name"]: c for c in temp}

    # -------------------------
    # Run search agent
    # -------------------------
    trace_box.write("Starting broader agent search...\n")

    with trace("Deterministic search flow") as t:
        # 1. Split query into structured subqueries
        trace_box.write("Decomposing query...\n")
        query_structure = await Runner.run(
            query_splitter_agent,
            query
        )

        # 2. Run the search agent using the structured form
        trace_box.write("Running search agent...\n")
        search_result = await Runner.run(
            search_agent,
            json.dumps(query_structure.final_output.model_dump(), indent=2)  # pass structured data
        )
        trace_box.write("Search agent completed.\n")

        college_names = search_result.final_output.college_names
        references = search_result.final_output.references

        # -------------------------
        # Embeddings for found college names
        # -------------------------
        trace_box.write("Embedding returned college names...\n")
        college_names_dict = {}

        for i, name_chunk in enumerate(chunk_list(college_names, BATCH_SIZE)):
            try:
                response = client.embeddings.create(
                    input=name_chunk, model=MODEL_NAME
                )
                for j, emb in enumerate(response.data):
                    college_names_dict[name_chunk[j]] = emb.embedding
            except Exception as e:
                trace_box.write(f"Embedding batch error: {e}\n")
                time.sleep(4)
                continue

        # -------------------------
        # Find top matches using cosine similarity
        # -------------------------
        trace_box.write("Computing similarities...\n")
        final_names = {}

        for query_name, emb in college_names_dict.items():
            first, second, third = (0, ""), (0, ""), (0, "")
            for stored_name, stored_embedding in embeddings_dict.items():
                score = cosine_similarity(emb, stored_embedding)
                if score > first[0]:
                    first = (score, stored_name)
                elif score > second[0]:
                    second = (score, stored_name)
                elif score > third[0]:
                    third = (score, stored_name)
            final_names[query_name] = [first[1], second[1], third[1]]

        # -------------------------
        # Translation Agent
        # -------------------------
        trace_box.write("Running translation agent...\n")
        final_colleges = await Runner.run(
            translation_agent,
            json.dumps(final_names, indent=2)
        )

    return (
        final_colleges.final_output.college_names,
        references,
        colleges,
        json.dumps(t.export(), indent=2)  # FULL agent trace
    )


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🎓 CollegeFinder AI — Multi-Agent Search")
st.write("Enter a natural language query and let the agents find matching U.S. colleges.")

query = st.text_input("Enter your college search query:", "")

run_button = st.button("Run Search")

if run_button and query.strip():
    placeholder = st.empty()
    trace_box = st.expander("🔍 Agent Trace (internal logs)", expanded=False)

    with st.spinner("Running search agents..."):
        results = asyncio.run(pipeline(query, trace_box))

    (
        final_college_names,
        references,
        colleges_dict,
        trace_output
    ) = results

    trace_box.write(trace_output)

    st.subheader("🏫 Matching Colleges")

    if not final_college_names:
        st.warning("No matching colleges found.")
    else:
        for name in final_college_names:
            if name in colleges_dict:
                desc = colleges_dict[name].get("description", "No description available.")
                state = colleges_dict[name].get("state", "N/A")
                city = colleges_dict[name].get("city", "N/A")
            else:
                desc = "(Description not found)"

            st.markdown(f"### **{name}**")
            st.markdown(f"### ***{state}, {city}***")
            st.write(desc)
            st.markdown("---")

    st.subheader("🔗 References")
    for ref in references:
        st.markdown(f"- [{ref}]({ref})")

