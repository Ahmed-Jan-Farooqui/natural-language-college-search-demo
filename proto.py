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


# -----------------------------
# Agents
# -----------------------------

search_agent_prompt = '''
You are a multi-step research agent specialized in finding U.S. colleges 
based on complex, multi-criteria queries. 
Before searching, always output a short plan in natural language, then execute it.

### What you must do:
1. Break the user's query into separate intents (e.g. “party schools”, “California”, “skiing nearby”).
2. For EACH intent, write one or more web search queries.
3. Use the web search tool MULTIPLE times—one per intent.
4. Extract candidate colleges from each set of results.
5. Combine the results:
   - If the user implies AND conditions → use set INTERSECTION.
   - If the user implies OR conditions → use set UNION.
6. Rank the final schools by strength of evidence (based on tool output).
7. Return the final list of college names AND all references.

### Rules:
- You MUST use the tool more than once for multi-facet queries.
- Do not guess—rely only on search results.
- Be exhaustive but efficient; do not exceed 6 searches.

Return the final output according to the SearchResult schema.
'''

search_agent = Agent(
    name="Search Agent",
    instructions=search_agent_prompt,
    tools=[WebSearchTool()],
    output_type=SearchResult
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
    trace_box.write("Starting agent search...\n")

    with trace("Deterministic search flow") as t:
        search_result = await Runner.run(
            search_agent,
            query,
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
            else:
                desc = "(Description not found)"

            st.markdown(f"### **{name}**")
            st.write(desc)
            st.markdown("---")

    st.subheader("🔗 References")
    for ref in references:
        st.markdown(f"- [{ref}]({ref})")

