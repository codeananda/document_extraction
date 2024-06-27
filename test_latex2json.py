import re
import json
import uuid
import os

# Function to extract citations within a given text
def extract_citations(text, references):
    citations = re.findall(r'\\cite[t|p]*\{([^}]+)\}', text)
    return list({citation for citation in citations if citation in references})  

# Function to create a section entry
def create_section_entry(section_id, section_title, content="", resources_cited=None):
    if resources_cited is None:
        resources_cited = []
    return {
        "section_id": section_id,
        "section": section_title,
        "content": content,
        "resources_cited_id": [i for i, ref in enumerate(references) if ref in resources_cited],
        "resources_cited_key": resources_cited,  # Keep the original keys as well for reference
    }

def clean_latex_content(content):
    # Remove lines starting with %
    content = '\n'.join(line for line in content.split('\n') if not line.strip().startswith('%'))
    
    # Remove unnecessary LaTeX commands
    content = re.sub(r'\\(usepackage|documentclass)\{.*?\}', '', content)
    
    # Remove LaTeX environments
    #content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', content, flags=re.DOTALL)
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    return content

def hierarchical_numbering(sections, method="counters"):
    section_counters = {"section": 0, "subsection": 0, "subsubsection": 0}
    hierarchy = []
    numbered_sections = []
    
    for section_type, section_title in sections:
        if method == "counters":
            if section_type == "section":
                section_counters["section"] += 1
                section_counters["subsection"] = 0
                section_counters["subsubsection"] = 0
                section_number = f"{section_counters['section']}"
            elif section_type == "subsection":
                section_counters["subsection"] += 1
                section_counters["subsubsection"] = 0
                section_number = f"{section_counters['section']}.{section_counters['subsection']}"
            elif section_type == "subsubsection":
                section_counters["subsubsection"] += 1
                section_number = f"{section_counters['section']}.{section_counters['subsection']}.{section_counters['subsubsection']}"
        
        elif method == "hierarchy":
            if section_type == "section":
                hierarchy = [section_title]
            elif section_type == "subsection":
                if len(hierarchy) > 1:
                    hierarchy = hierarchy[:2]
                hierarchy.append(section_title)
            elif section_type == "subsubsection":
                if len(hierarchy) > 2:
                    hierarchy = hierarchy[:3]
                hierarchy.append(section_title)
            section_number = '.'.join(map(str, range(1, len(hierarchy) + 1)))

        elif method == "combined":
            if section_type == "section":
                section_counters["section"] += 1
                section_counters["subsection"] = 0
                section_counters["subsubsection"] = 0
                section_number = str(section_counters["section"])
            elif section_type == "subsection":
                section_counters["subsection"] += 1
                section_counters["subsubsection"] = 0
                section_number = f"{section_counters['section']}.{section_counters['subsection']}"
            elif section_type == "subsubsection":
                section_counters["subsubsection"] += 1
                section_number = f"{section_counters['section']}.{section_counters['subsection']}.{section_counters['subsubsection']}"
            section_title = f"{section_number} {section_title}"

        numbered_sections.append((section_type, f"{section_number} {section_title}"))
    
    return numbered_sections

# Function to extract resource information from BibTeX
def extract_resources(bibtex_content):
    resources = []
    entries = re.findall(r'@(\w+)\{([^,]+),(.+?)\n\}', bibtex_content, re.DOTALL)
    for i, (entry_type, citation_key, content) in enumerate(entries):
        title_match = re.search(r'title\s*=\s*\{(.+?)\}', content, re.DOTALL)
        author_match = re.search(r'author\s*=\s*\{(.+?)\}', content, re.DOTALL)
        year_match = re.search(r'year\s*=\s*\{(.+?)\}', content)
        url_match = re.search(r'url\s*=\s*\{(.+?)\}', content)
        doi_match = re.search(r'doi\s*=\s*\{(.+?)\}', content)
        
        title = title_match.group(1).strip() if title_match else None
        author = author_match.group(1).strip() if author_match else None
        year = year_match.group(1).strip() if year_match else None
        url = url_match.group(1).strip() if url_match else doi_match.group(1).strip() if doi_match else None
        
        resources.append({
            "resource_id": i + 1,
            "resource_key": citation_key.strip(),
            "description": title if title else "" + "\nAuthor:"+author if author else "" + "\nYear:"+year if year else "",
            "url": url
        })
    return resources

# Directory containing the LaTeX and BibTeX files
data_dir = './data/latex'

# List all .tex files in the data directory
tex_files = [f for f in os.listdir(data_dir) if f.endswith('.tex')]

for tex_file in tex_files:
    # Construct the corresponding BibTeX file name
    bib_file = tex_file.replace('.tex', '.bib')

    # Read the LaTeX file
    with open(os.path.join(data_dir, tex_file), 'r') as file:
        latex_content = file.read()

    # Read the BibTeX file
    with open(os.path.join(data_dir, bib_file), 'r') as file:
        bibtex_content = file.read()

    latex_content = clean_latex_content(latex_content)

    # Extract title
    title_match = re.search(r'\\title\{(.+?)\}', latex_content, re.DOTALL)
    title = title_match.group(1).strip() if title_match else 'No Title Found'

    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', latex_content, re.DOTALL | re.IGNORECASE) or \
                     re.search(r'\\abstract\{(.+?)\}', latex_content, re.DOTALL) or \
                     re.search(r'\\abst[ \n](.*?)\\xabst', latex_content, re.DOTALL) or \
                     re.search(r'\\section\*\{abstract\}(.+?)(?=\\section|\Z)', latex_content, re.DOTALL | re.IGNORECASE)
    abstract = abstract_match.group(1).strip() if abstract_match else 'No Abstract Found'

    # Extract sections, subsections, and subsubsections
    section_pattern = r'\\((?:sub)*section)\{(.+?)\}'
    sections = re.findall(section_pattern, latex_content)
    numbered_sections = hierarchical_numbering(sections, method="counters")  # Change "counters" to "hierarchy" or "combined" as needed

    # Extract all references from BibTeX BEFORE processing sections
    references = re.findall(r'@.*?\{(.*?),', bibtex_content, re.DOTALL)

    plan = []
    section_id = 0
    cited_references = set()  # Set to track actually cited references

    for section_type, section_title in numbered_sections:
        section_id += 1
        
        # Extract the content of the section
        original_section_title = section_title.split(" ", 1)[1]  # Get the original section title without numbering
        section_regex = rf'\\{section_type}\{{{re.escape(original_section_title)}\}}(.*?)(?=\\(?:sub)*section\{{|\\end\{{document\}})'
        section_content_match = re.search(section_regex, latex_content, re.DOTALL)
        content = section_content_match.group(1).strip() if section_content_match else ''
        
        # Extract citations within the section
        resources_cited = extract_citations(content, references)
        cited_references.update(resources_cited)  # Add cited references to the set
        
        plan.append(create_section_entry(section_id, section_title, content, resources_cited))

    # Extract references from the BibTeX file
    resources = extract_resources(bibtex_content)

    # Filter resources to include only those actually cited in the text
    resources = [res for res in resources if res['resource_key'] in cited_references]

    # Create the JSON structure
    paper_data = {
        "id": str(uuid.uuid4()),
        "title": title,
        "abstract": abstract,
        "plan": plan,
        "resources": resources
    }

    # Convert to JSON
    json_output = json.dumps(paper_data, indent=2)
    # Regex replace to remove any newline between arrays brackets [ and ] in resources_cited_key and resources_cited_id in the JSON output
    json_output = re.sub(r'("resources_cited_(id|key)?"\s*:\s*\[\n\s*([^]]+?)\s*\])', lambda m: m.group(0).replace('\n', '').replace(' ', ''), json_output)

    # Save the JSON output to a file in the output folder (check if folder exists or create it before)
    output_dir = './output/latex'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_filename = os.path.join(output_dir, tex_file.replace('.tex', '.json'))
    with open(json_filename, 'w') as file:
        file.write(json_output)

    # Display the top 50 lines of JSON
    print(json_output[:500])