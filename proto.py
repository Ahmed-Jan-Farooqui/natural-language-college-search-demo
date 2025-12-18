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
    queries_ran: list[str]


class TranslationResult(BaseModel):
    college_names: list[str]



# -----------------------------
# Agents
# -----------------------------

search_agent_prompt = """
    You are a recall-first college research agent.

    Your goal is to return AS MANY potentially relevant U.S. colleges as possible.

    You will be given a single natural-language query from a user.
    You must decide how to search — there is NO predefined structure.

    ### Core principles
    - Recall > precision
    - Include borderline, indirect, and partial matches
    - Never over-filter
    - Redundancy is good
    - Prioritize sources that provide a list of colleges relevant to the query.

    ### Execution steps

    1. Rewrite the user query into MULTIPLE diverse web searches, including:
    - Broad interpretations
    - Narrow interpretations
    - Synonyms and rephrasings
    - Location-based variants
    - Program-based variants
    - Lifestyle or attribute-based variants

    2. You MUST perform AT LEAST 3 distinct web searches.
    - More if the query is complex
    - Each search should target a different angle

    3. For EACH search:
    - Extract ALL colleges mentioned
    - Include schools even if relevance is uncertain
    - Prefer inclusion over exclusion

    4. Maintain an internal mapping of:
    college_name → number of mentions / search angles

    5. Rank colleges by:
    - Frequency of appearance
    - Strength of association (if evident)
    - But NEVER drop low-ranked colleges unless clearly irrelevant

    6. Do NOT try to enforce logical constraints.
    The pipeline downstream will handle normalization and filtering.

    ### Output
    Return:
    - A ranked list of college names (highest recall first)
    - A list of all references used
    - A list of all queries that you ran.

    Return output using the SearchResult schema.
"""



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
    
    data: {
        <college_name>: <college_id>
        .
        .
        .    
    }
    colleges: [<college_name_1>, ..., <college_name_n>]


    Your job is to map each college's name in the colleges list to a college in the available data dictionary.
    and return the final, mapped names.
    If there are no matches found for a given college in the data, skip it. 
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
        college_ids = {c["name"]: c["college_id"] for c in temp}

    # -------------------------
    # Run search agent
    # -------------------------
    trace_box.write("Starting broader agent search...\n")

    with trace("Deterministic search flow") as t:
        # 2. Run scoring-based search agent
        search_result = await Runner.run(
            search_agent,
            query,
        )

        college_names = search_result.final_output.college_names
        references = search_result.final_output.references
        searches = search_result.final_output.queries_ran

        trace_box.write(f'Search returned {len(college_names)} results...\n')
        trace_box.write(f'College names:\n{college_names}')

        # # -------------------------
        # # Embeddings for found college names
        # # -------------------------
        # trace_box.write("Embedding returned college names...\n")
        # college_names_dict = {}

        # for i, name_chunk in enumerate(chunk_list(college_names, BATCH_SIZE)):
        #     try:
        #         response = client.embeddings.create(
        #             input=name_chunk, model=MODEL_NAME
        #         )
        #         for j, emb in enumerate(response.data):
        #             college_names_dict[name_chunk[j]] = emb.embedding
        #     except Exception as e:
        #         trace_box.write(f"Embedding batch error: {e}\n")
        #         time.sleep(4)
        #         continue

        # # -------------------------
        # # Find top matches using cosine similarity
        # # -------------------------
        # trace_box.write("Computing similarities...\n")
        # final_names = {}

        # for query_name, emb in college_names_dict.items():
        #     first, second, third = (0, ""), (0, ""), (0, "")
        #     for stored_name, stored_embedding in embeddings_dict.items():
        #         score = cosine_similarity(emb, stored_embedding)
        #         if score > first[0]:
        #             first = (score, stored_name)
        #         elif score > second[0]:
        #             second = (score, stored_name)
        #         elif score > third[0]:
        #             third = (score, stored_name)
        #     final_names[query_name] = [first[1], second[1], third[1]]

        # -------------------------
        # Translation Agent
        # -------------------------
        trace_box.write("Running translation agent...\n")
        trans_query = f'Available colleges: {json.dumps(college_ids, indent=2)} \n Colleges to match: {college_names}'
        trans_result = await Runner.run(
            translation_agent,
            trans_query,
        )
        final_colleges = list(set(trans_result.final_output.college_names))
        

    return (
        final_colleges,
        references,
        colleges,
        json.dumps(t.export(), indent=2),  # FULL agent trace
        searches,
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
        trace_output,
        searches,
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
                college_id = colleges_dict[name].get("college_id")
                url = f"https://app.kollegio.ai/college/{college_id}"
            else:
                st.markdown(f'### **{name}**')
                st.markdown(f'### * Found in search, but mistranslated / does not exist in our database.')
                continue

            st.markdown(f"### **[{name}]({url})**")
            st.markdown(f"### ***{state}, {city}***")
            st.write(desc)
            st.markdown("---")

    st.subheader("🔗 References")
    for ref in references:
        st.markdown(f"- [{ref}]({ref})")

    st.subheader("🔗 Searches")
    for search in searches:
        st.markdown(f"- {search} ")

