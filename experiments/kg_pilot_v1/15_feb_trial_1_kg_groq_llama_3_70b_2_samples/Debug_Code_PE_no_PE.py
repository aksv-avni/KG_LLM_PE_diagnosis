# =============== Debug Code for 2 samples analysis : PE and no PE ======================

import os
import json
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
from groq import Groq
from google.colab import userdata

# 1. SETUP PIPELINE (Ensure en_core_sci_sm is installed)
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

# Note: Set your GROQ_API_KEY as an environment variable or replace here
# Using the API key directly from the previously noted value to fix the GroqError
client = Groq(api_key="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")

def analyze_sample(sample_id, text):
    print(f"\n{'='*30}")
    print(f"ANALYZING SAMPLE ID: {sample_id}")
    print(f"{'='*30}")

    # --- STEP 1: SCISPACY NER & LINKING ---
    print("\n[STEP 1] ScispaCy: Entity Extraction & UMLS Linking")
    doc = nlp(text)
    print(f"nlp(text) : {doc} ")
      
    # Abbreviation Expansion
    expanded_text = text
    for abrv in doc._.abbreviations:
        expanded_text = expanded_text.replace(str(abrv), str(abrv._.long_form))
    print(f"Expanded Text: {expanded_text}")

    detected_entities = []
    for ent in doc.ents:
        cui = ent._.kb_ents[0][0] if ent._.kb_ents else "Unknown"
        canonical_name = linker.kb.cui_to_entity[cui].canonical_name if cui != "Unknown" else "N/A"
        detected_entities.append({"text": ent.text, "cui": cui, "canonical": canonical_name})
        
        print(f" - Found: '{ent.text}' -> CUI: {cui} ({canonical_name})")
    print(f"detected entities list : {detected_entities}")    
    print(f"DEBUG (ScispaCy Entities): {json.dumps(detected_entities, indent=2)}") # Debug line added

    # --- STEP 2: LLM TRIPLET EXTRACTION ---
    print("\n[STEP 2] LLM: Triplet Extraction (Groq/Llama-3)")
    sys_instruct = (
        "You are a medical data architect. Extract knowledge triplets as JSON. "
        "Structure: {'triplets': [{'subject': 'Anatomy', 'predicate': 'Relation', 'object': 'Pathology/Assertion'}]} "
        "Use relations: located_in, manifested_as, associated_with."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": sys_instruct},
                {"role": "user", "content": f"Extract triplets from: {expanded_text}"}
            ],
            response_format={"type": "json_object"}
        )
        llm_raw = json.loads(response.choices[0].message.content)
        print("Raw LLM Output:")
        print(json.dumps(llm_raw, indent=2))
    except Exception as e:
        print(f"Error calling Groq: {e}")
        llm_raw = {"triplets": []}

    # --- STEP 3: MAPPING / GROUNDING ---
    print("\n[STEP 3] Grounding: Mapping LLM Output to UMLS")
    final_grounded_triplets = []
    for tri in llm_raw.get('triplets', []):
        # Ground Subject
        subj_doc = nlp(tri['subject'])
        subj_cui = subj_doc.ents[0]._.kb_ents[0][0] if (subj_doc.ents and subj_doc.ents[0]._.kb_ents) else "Unknown"

        # Ground Object
        obj_doc = nlp(tri['object'])
        obj_cui = obj_doc.ents[0]._.kb_ents[0][0] if (obj_doc.ents and obj_doc.ents[0]._.kb_ents) else "Unknown"

        tri_data = {
            "triplet": tri,
            "grounded_subject_cui": subj_cui,
            "grounded_object_cui": obj_cui
        }
        final_grounded_triplets.append(tri_data)
        print(f" - Triplet: ({tri['subject']}) --[{tri['predicate']}]--> ({tri['object']})")
        print(f"   Mapped: ({subj_cui}) -> ({obj_cui})")
    print(f"final grounded triplet list : {final_grounded_triplets}")
    print(f"DEBUG (Final Grounded Triplets): {json.dumps(final_grounded_triplets, indent=2)}") # Debug line added
    return final_grounded_triplets

# TEST SAMPLES
pe_positive = ": 1.  Multiple pulmonary emboli in the right upper, right lower, and  left lower lobes. No evidence on CT of right heart strain. 2.  Worsening groundglass and consolidative opacities in the lungs  bilaterally, most marked in the upper lobes, concerning for worsening  multifocal infection. Selected areas of peripheral  groundglass/consolidation as related to superimposed pulmonary  infarct cannot be excluded. Above findings were communicated to <HCW> by <HCW> at <TIME> on  <DATE>."
pe_negative = "IMPRESSION: 1.NO EVIDENCE OF PULMONARY EMBOLISM. 2.SMALL BILATERAL PLEURAL EFFUSIONS WITH ASSOCIATED ATELECTASIS,  WORSE IN THE RIGHT LOWER LOBE. 3.STATUS POST CHOLECYSTECTOMY WITH A SMALL AMOUNT OF FLUID IN THE  GALLBLADDER FOSSA, AND DILATED COMMON BILE DUCT WITH WALL ENHANCEMENT. SUMMARY: 4-POSSIBLE SIGNIFICANT FINDING, MAY NEED ACTION This dictation is for the dedicated CT PE study.  Please refer to the  separate CT abdomen and pelvis dictation for those findings."

# Execute
analyze_sample("PE_POS_11561", pe_positive)
analyze_sample("PE_NEG_647", pe_negative)

pe_negative_results = [
  {
    "triplet": {
      "subject": "Pleural Effusions",
      "predicate": "located_in",
      "object": "Bilateral Lungs"
    },
    "grounded_subject_cui": "C0032227",
    "grounded_object_cui": "C0238767"
  },
  {
    "triplet": {
      "subject": "Atelectasis",
      "predicate": "associated_with",
      "object": "Pleural Effusions"
    },
    "grounded_subject_cui": "C0004144",
    "grounded_object_cui": "C0032227"
  },
  {
    "triplet": {
      "subject": "Pleural Effusions",
      "predicate": "manifested_as",
      "object": "Fluid in Pleural Space"
    },
    "grounded_subject_cui": "C0032227",
    "grounded_object_cui": "C0302908"
  },
  {
    "triplet": {
      "subject": "Dilated Common Bile Duct",
      "predicate": "associated_with",
      "object": "Wall Enhancement"
    },
    "grounded_subject_cui": "C0009437",
    "grounded_object_cui": "C2349975"
  },
  {
    "triplet": {
      "subject": "Fluid",
      "predicate": "located_in",
      "object": "Gallbladder Fossa"
    },
    "grounded_subject_cui": "C0302908",
    "grounded_object_cui": "C0227511"
  },
  {
    "triplet": {
      "subject": "Cholecystectomy",
      "predicate": "associated_with",
      "object": "Status Post Surgery"
    },
    "grounded_subject_cui": "C0008320",
    "grounded_object_cui": "C0241311"
  }
]

pe_positive_results = [
  {
    "triplet": {
      "subject": "Pulmonary emboli",
      "predicate": "located_in",
      "object": "Right upper, right lower, and left lower lobes"
    },
    "grounded_subject_cui": "C0034065",
    "grounded_object_cui": "C1261074"
  },
  {
    "triplet": {
      "subject": "Groundglass and consolidative opacities",
      "predicate": "manifested_as",
      "object": "Worsening multifocal infection"
    },
    "grounded_subject_cui": "C0598786",
    "grounded_object_cui": "C0332271"
  },
  {
    "triplet": {
      "subject": "Peripheral groundglass/consolidation",
      "predicate": "associated_with",
      "object": "Superimposed pulmonary infarct"
    },
    "grounded_subject_cui": "C0205100",
    "grounded_object_cui": "C0034074"
  }
]

!pip install pyvis

import networkx as nx
from pyvis.network import Network
import pandas as pd
import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker



# Assuming pe_positive_results and pe_negative_results are already populated
# from the previous execution of analyze_sample in cell wjP7RbD1_kG6.
# If this cell is run independently, those variables would need to be re-assigned.

all_triplets_dynamic = []

# Process PE Positive results
for entry in pe_positive_results:
    subj_text = entry['triplet']['subject']
    obj_text = entry['triplet']['object']
    pred = entry['triplet']['predicate']
    sample_id = "11561"

    subj_cui = entry['grounded_subject_cui']
    obj_cui = entry['grounded_object_cui']

    # Use canonical names from linker.kb if CUI is known, else original text
    subj_name = linker.kb.cui_to_entity[subj_cui].canonical_name if subj_cui != 'Unknown' and subj_cui in linker.kb.cui_to_entity else subj_text
    obj_name = linker.kb.cui_to_entity[obj_cui].canonical_name if obj_cui != 'Unknown' and obj_cui in linker.kb.cui_to_entity else obj_text

    all_triplets_dynamic.append({
        "subject": subj_name,
        "predicate": pred,
        "object": obj_name,
        "sample_id": sample_id
    })

# Process PE Negative results
for entry in pe_negative_results:
    subj_text = entry['triplet']['subject']
    obj_text = entry['triplet']['object']
    pred = entry['triplet']['predicate']
    sample_id = "647"

    subj_cui = entry['grounded_subject_cui']
    obj_cui = entry['grounded_object_cui']

    # Use canonical names from linker.kb if CUI is known, else original text
    subj_name = linker.kb.cui_to_entity[subj_cui].canonical_name if subj_cui != 'Unknown' and subj_cui in linker.kb.cui_to_entity else subj_text
    obj_name = linker.kb.cui_to_entity[obj_cui].canonical_name if obj_cui != 'Unknown' and obj_cui in linker.kb.cui_to_entity else obj_text

    all_triplets_dynamic.append({
        "subject": subj_name,
        "predicate": pred,
        "object": obj_name,
        "sample_id": sample_id
    })

def visualize_mini_kg(triplets, output_file="mini_kg_analysis.html"):
    # Initialize NetworkX MultiDiGraph (to allow multiple edges if needed)
    G = nx.MultiDiGraph()

    for tri in triplets:
        # Add edge with sample_id metadata
        G.add_edge(
            tri['subject'],
            tri['object'],
            label=tri['predicate'],
            title=f"Sample ID: {tri['sample_id']}"
        )

    # Initialize Pyvis
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)

    # Styling Nodes based on Clinical Categories
    for node in net.nodes:
        label = str(node['id']).lower() # Ensure label is a string for .lower()
        # Anatomy (Green)
        if any(x in label for x in ["lobe", "bilateral", "lung", "heart", "artery", "vein", "trachea", "bronchus", "diaphragm", "chest", "liver", "kidney", "spleen", "abdomen", "pelvis", "brain", "spine", "bone", "fossa", "duct", "tissue"]):
            node['color'] = "#90EE90"
        # Assertions/Status (Yellow)
        elif any(x in label for x in ["present", "absent", "no evidence", "possible", "rule out", "confirmed", "status", "worsening", "small", "dilated"]):
            node['color'] = "#F0E68C"
        # Pathology/Condition (Salmon)
        else:
            node['color'] = "#FFA07A"

    # Physics configuration for better layout of small clusters
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "solver": "forceAtlas2Based"
      }
    }
    """
    )

    net.save_graph(output_file)
    print(f"Mini-KG saved to {output_file}")

visualize_mini_kg(all_triplets_dynamic)

import pandas as pd

def export_pe_triplets_to_csv(positive_results, negative_results, output_file="pe_analysis_triplets.csv"):
    all_records = []

    # Process positive results
    for entry in positive_results:
        all_records.append({
            "sample_id": "11561",
            "subject_text": entry['triplet']['subject'],
            "predicate": entry['triplet']['predicate'],
            "object_text": entry['triplet']['object'],
            "grounded_subject_cui": entry['grounded_subject_cui'],
            "grounded_object_cui": entry['grounded_object_cui']
        })

    # Process negative results
    for entry in negative_results:
        all_records.append({
            "sample_id": "647",
            "subject_text": entry['triplet']['subject'],
            "predicate": entry['triplet']['predicate'],
            "object_text": entry['triplet']['object'],
            "grounded_subject_cui": entry['grounded_subject_cui'],
            "grounded_object_cui": entry['grounded_object_cui']
        })

    df_triplets = pd.DataFrame(all_records)
    df_triplets.to_csv(output_file, index=False)
    print(f"✅ Exported {len(df_triplets)} triplets to {output_file}")
    return df_triplets

# Execute the export function
df_pe_triplets_analysis = export_pe_triplets_to_csv(pe_positive_results, pe_negative_results)

# Display the first few rows of the DataFrame
display(df_pe_triplets_analysis.head())


import spacy
import scispacy
from scispacy.linking import EntityLinker

# 1. Load the model and linker (if not already in your script)
# Use 'en_core_sci_sm' or whichever scispacy model you are currently using
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# Access the knowledge base (kb) from the linker
linker = nlp.get_pipe("scispacy_linker")

def get_canonical_name_from_cui(cui):
    """
    Given a UMLS CUI, return its canonical name using scispacy's KB.
    """
    try:
        # linker.kb.cui_to_entity is a dictionary where keys are CUIs
        entity = linker.kb.cui_to_entity.get(cui)
        if entity:
            return entity.canonical_name
        return "Unknown CUI"
    except Exception as e:
        return f"Error: {str(e)}"

# --- EXAMPLE USAGE ---

# Suppose these are the CUIs you obtained from your pipeline
cui_list = [
    'C0032227',
    'C0238767',
    'C0004144',
    'C0032227',
    'C0032227',
    'C0302908',
    'C0009437',
    'C2349975',
    'C0302908',
    'C0227511',
    'C0008320',
    'C0241311'
]


print(f"{'CUI':<12} | {'Canonical Name'}")
print("-" * 40)

for cui in cui_list:
    name = get_canonical_name_from_cui(cui)
    print(f"{cui:<12} | {name}")

# --- INTEGRATING INTO YOUR GRAPH ---
# If you want to rename nodes in your NetworkX graph G from CUIs to Names:
# mapping = {node: get_canonical_name_from_cui(node) for node in G.nodes()}
# G_named = nx.relabel_nodes(G, mapping)

# CUI          | Canonical Name
# ----------------------------------------
# C0032227     | Pleural effusion disorder
# C0238767     | Bilateral
# C0004144     | Atelectasis
# C0032227     | Pleural effusion disorder
# C0032227     | Pleural effusion disorder
# C0302908     | Liquid substance
# C0009437     | Common bile duct structure
# C2349975     | Enhance (action)
# C0302908     | Liquid substance
# C0227511     | Structure of gallbladder fossa of liver
# C0008320     | Cholecystectomy procedure
# C0241311     | post operative (finding)