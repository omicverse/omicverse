from .._registry import register_function

@register_function(
    aliases=["本地GPT注释", "gptcelltype_local", "local_ai_celltype", "本地AI注释", "离线细胞注释"],
    category="single",
    description="Local LLM-powered cell type annotation using Hugging Face models without API requirements",
    examples=[
        "# Basic local LLM annotation",
        "markers = ov.single.get_celltype_marker(adata, clustertype='leiden')",
        "result = ov.single.gptcelltype_local(markers, tissuename='PBMC',",
        "                                     model_name='Qwen/Qwen2-7B-Instruct')",
        "# Using different local model",
        "result = ov.single.gptcelltype_local(markers, tissuename='Brain',",
        "                                     model_name='microsoft/DialoGPT-medium')",
        "# Custom top genes and species",
        "result = ov.single.gptcelltype_local(markers, tissuename='Liver',",
        "                                     speciename='mouse', topgenenumber=5)",
        "# Apply results to AnnData",
        "adata.obs['local_gpt_celltype'] = adata.obs['leiden'].map(result)",
        "# Compare with online GPT results",
        "ov.pl.embedding(adata, color=['local_gpt_celltype', 'gpt_celltype'])"
    ],
    related=["single.gptcelltype", "single.get_celltype_marker", "single.pySCSA"]
)
def gptcelltype_local(input, tissuename=None, speciename='human',
                model_name='Qwen/Qwen2-7B-Instruct', topgenenumber=10):
    """Annotate cell types with a local instruction-tuned LLM.

    Parameters
    ----------
    input : dict or pandas.DataFrame
        Marker definition per cluster. Accepts either a
        ``dict[cluster_id -> list[str]]`` of marker genes, or a
        DE table with ``cluster``, ``names``, ``logfoldchanges`` columns.
    tissuename : str or None, default=None
        Tissue context included in the prompt.
    speciename : str, default='human'
        Species context string included in the prompt.
    model_name : str, default='Qwen/Qwen2-7B-Instruct'
        Local Hugging Face model name or path used for generation.
    topgenenumber : int, default=10
        Maximum number of marker genes retained per cluster.

    Returns
    -------
    dict
        Mapping from cluster ID to predicted cell type name.

    Examples
    --------
    >>> markers = {"0": ["CD3D", "IL7R"], "1": ["NKG7", "GNLY"]}
    >>> res = gptcelltype_local(markers, tissuename="PBMC", speciename="human")
    """
    import re
    import numpy as np
    import pandas as pd
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    
    # Load the model and tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map='cuda', 
        torch_dtype='auto', 
        trust_remote_code=True
    )
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")
    
    message_template = (
        f"Identify cell types of {tissuename} cells in {speciename} using the above markers separately for each row.\n"
        "Provide the cell type name, followed by the reason, which should be enclosed in square brackets.\n"
        "Some can be a mixture of multiple cell types. If so, seperate them with semicolon.\n\n"
        "Output format:\n"
        "cluster: cell type [marker(s)]\n"
        "Output example:\n"
        "0: T cells [CD3D, IL7R]\n"
        "1: Cytotoxic T cells [CCL5, NKG7, GZMA]; Natural Killer (NK) cells [GNLY, KLRD1]"
    )
    
    allres = {}
    cutnum = int(np.ceil(len(input) / 30))
    if cutnum > 1:
        cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
    else:
        cid = np.ones(len(input), dtype=int)
    
    for i in range(1, cutnum + 1):
        id_list = [j for j, x in enumerate(cid) if x == i]
        message = '\n'.join(
            [f"{k}: {v}" for k, v in input.items() if list(input.keys()).index(k) in id_list]
        ) + '\n\n' + message_template

        messages = [
            {"role": "system", "content": "You are an experienced biologist with particular expertise in molecular biology, cell biology, and bioinformatics."},
            {"role": "user", "content": message},
        ]

        generation_args = {
            "max_new_tokens": 5000,
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": False,
        }
        generated = pipe(messages, **generation_args)
        print(generated[0]['generated_text'])

        pattern = r'\d+:\s+(.+?)\s+\[.*?\]'
        res = re.findall(pattern, generated[0]['generated_text'])
        for idx, cell_type in zip(id_list, res):
            key = list(input.keys())[idx]
            allres[key] = 'unknown' if input[key] == 'unknown' else cell_type

    print('Note: It is always recommended to check the results returned by the LLM in case of AI hallucination, before going to downstream analysis.')
    return allres
