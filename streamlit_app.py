import tempfile
from pathlib import Path
import streamlit as st
from karyo_ai.pipeline import analyze_image

st.set_page_config(page_title='KaryoAI research review')
st.title('KaryoAI — Draft karyogram review')
st.error('RESEARCH/WORKFLOW-ASSISTANCE ONLY. Not for clinical diagnosis. A certified cytogeneticist must review every output; formal validation is required before clinical/SOP use.')
upload=st.file_uploader('Upload chromosome image',type=['png','jpg','jpeg','tif','tiff'])
species=st.selectbox('Species',['human','mouse']); modality=st.selectbox('Stain/modality',['auto','giemsa','fluorescent'])
reviewer=st.text_input('Reviewer name'); approve=st.checkbox('I have manually reviewed the draft; mark this export reviewed')
if upload and st.button('Create draft analysis'):
    suffix=Path(upload.name).suffix
    with tempfile.TemporaryDirectory() as temp:
        source=Path(temp)/f'upload{suffix}'; source.write_bytes(upload.getvalue()); out=Path(temp)/'output'
        result=analyze_image(str(source),str(out),species,modality,{},reviewer,approve)
        st.image(str(out/f'{source.stem}_annotated.png'),caption='Candidate-object annotation')
        st.image(str(out/f'{source.stem}_draft_karyogram.png'),caption='Draft layout — verify labels and pairing')
        st.dataframe([o.to_dict() for o in result.objects])
        st.download_button('Download JSON', (out/f'{source.stem}_result.json').read_bytes(), file_name='karyo_result.json')
        st.download_button('Download report', (out/f'{source.stem}_report.md').read_bytes(), file_name='karyo_report.md')
