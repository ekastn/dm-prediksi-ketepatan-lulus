import pandas as pd

try:
    df = pd.read_excel('references/Literature_Review_Performance_Interpretability_decission_tree.xlsx')
    with open('references/references_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Literature Review Details\n\n')
        for idx, row in df.iterrows():
            f.write(f'### Reference {row["ID"]}: {row["Full Title"]}\n')
            f.write(f'- **Authors:** {row["Authors"]}\n')
            f.write(f'- **Year:** {row["Year"]} | **Journal:** {row["Journal"]} ({row["Scopus Quartile"]}) | **Publisher:** {row["Publisher"]}\n')
            f.write(f'- **DOI:** {row["DOI"]}\n')
            f.write(f'- **Thematic Category:** {row["Thematic Category"]}\n')
            f.write(f'- **Research Objective:** {row["Research Objective"]}\n')
            f.write(f'- **Dataset:** {row["Dataset"]}\n')
            f.write(f'- **Algorithms:** {row["Machine Learning Algorithm"]}\n')
            f.write(f'- **Main Findings:** {row["Main Findings"]}\n')
            f.write(f'- **Relevance:** {row["Relevance to Study"]}\n')
            f.write(f'- **Section to Cite:** {row["Section to Cite"]}\n')
            f.write(f'- **IEEE Reference:** {row["IEEE Reference"]}\n')
            f.write(f'- **PDF:** {row["PDF File"]}\n\n')
            f.write('---\n\n')
    print('Markdown summary generated successfully!')
except Exception as e:
    print(f'Error: {e}')
