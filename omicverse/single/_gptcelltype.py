import requests
import os
import numpy as np
import pandas as pd
from .._registry import register_function

@register_function(
    aliases=["GPT细胞类型注释", "gptcelltype", "ai_celltype", "GPT注释", "AI细胞注释"],
    category="single",
    description="AI-powered cell type annotation using GPT-4, Qwen, Kimi, and other large language models",
    prerequisites={
        'functions': ['leiden', 'get_celltype_marker']
    },
    requires={
        'obs': []  # Dynamic: requires clustertype column
    },
    produces={
        'obs': []  # User manually maps result to obs
    },
    auto_fix='escalate',
    examples=[
        "# Basic GPT annotation with Qwen",
        "os.environ['AGI_API_KEY'] = 'your-api-key'",
        "markers = ov.single.get_celltype_marker(adata, clustertype='leiden')",
        "result = ov.single.gptcelltype(markers, tissuename='PBMC',",
        "                               speciename='human', provider='qwen')",
        "# Using OpenAI GPT-4",
        "result = ov.single.gptcelltype(markers, tissuename='Brain',",
        "                               provider='openai', model='gpt-4o')",
        "# Using Kimi (Moonshot)",
        "result = ov.single.gptcelltype(markers, tissuename='Blood',",
        "                               provider='kimi', model='moonshot-v1-8k')",
        "# Custom model with base_url",
        "result = ov.single.gptcelltype(markers, tissuename='Liver',",
        "                               model='custom-model',",
        "                               base_url='https://api.example.com/v1')",
        "# Apply results to AnnData",
        "adata.obs['gpt_celltype'] = adata.obs['leiden'].map(result)"
    ],
    related=["single.get_celltype_marker", "single.gptcelltype_local", "single.pySCSA"]
)
def gptcelltype(input, tissuename=None, speciename='human',
                provider='qwen',model='qwen-plus', topgenenumber=10,
                base_url=None):
    r"""Annotate cluster cell types with a remote LLM service.

    Parameters
    ----------
    input : dict or pandas.DataFrame
        Cluster marker input. Use either a
        ``dict[cluster_id -> list[str]]`` of marker genes, or a
        DE result table containing ``cluster``, ``names``, and
        ``logfoldchanges`` columns.
    tissuename : str or None, default=None
        Tissue context provided to the model prompt (for example, PBMC or brain).
    speciename : str, default='human'
        Species context string included in the prompt.
    provider : str, default='qwen'
        LLM provider preset used to infer default API endpoint.
        Supported values are ``'openai'``, ``'kimi'``, and ``'qwen'``.
    model : str, default='qwen-plus'
        Chat-completion model name used by the selected provider.
    topgenenumber : int, default=10
        Number of top marker genes retained per cluster before prompting.
    base_url : str or None, default=None
        Custom chat-completion endpoint base URL. If ``None``, it is selected
        from ``provider``.

    Returns
    -------
    dict or str
        When ``AGI_API_KEY`` is available, returns ``dict[cluster_id -> celltype]``.
        Otherwise returns the generated prompt text for manual use.

    Examples
    --------
    >>> markers = {"0": ["MS4A1", "CD79A"], "1": ["NKG7", "PRF1"]}
    >>> res = gptcelltype(markers, tissuename="PBMC", speciename="human")
    >>> adata.obs["gpt_celltype"] = adata.obs["leiden"].map(res)
    """
    from openai import OpenAI
    import os
    import numpy as np
    import pandas as pd
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1/"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    if QWEN_API_KEY == "":
        print("Note: AGI API key not found: returning the prompt itself.")
        API_flag = False
    else:
        API_flag = True
    client = OpenAI(
        api_key=QWEN_API_KEY, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url=base_url,
    )
    
    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")
    
    if not API_flag:
        message = f'Identify cell types of {tissuename} cells in {speciename} using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([f'{k}: {v}' for k, v in input.items()])
        return message
    else:
        print("Note: AGI API key found: returning the cell type annotations.")
        cutnum = int(np.ceil(len(input) / 30))
        if cutnum > 1:
            cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
        else:
            cid = np.ones(len(input), dtype=int)
        
        allres = {}
        for i in range(1, cutnum + 1):
            id_list = [j for j, x in enumerate(cid) if x == i]
            flag = False
            while not flag:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", 
                               "content": f'Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([input[list(input.keys())[j]] for j in id_list if input[list(input.keys())[j]] != 'unknown'])}]
                )
                #return response
                res = response.choices[0].message.content.split('\n')
                if len(res) == len(id_list):
                    flag = True
            for idx, cell_type in zip(id_list, res):
                key = list(input.keys())[idx]
                allres[key] = 'unknown' if input[key] == 'unknown' else cell_type.strip(',').strip()
        
        
        print('Note: It is always recommended to check the results returned by GPT-4 in case of AI hallucination, before going to downstream analysis.')
        return allres
    


def gpt4celltype(input_data, tissuename=None, speciename='human',
                provider='qwen', model='qwen-plus', topgenenumber=10,
                base_url=None):
    r"""Annotate cell types with batched HTTP requests to chat-completion APIs.

    Parameters
    ----------
    input_data : dict or pandas.DataFrame
        Marker definition per cluster. Same accepted formats as
        :func:`gptcelltype`.
    tissuename : str or None, default=None
        Tissue context used in prompts.
    speciename : str, default='human'
        Species context used in prompts.
    provider : str, default='qwen'
        Provider preset used to infer a default ``base_url``.
    model : str, default='qwen-plus'
        Model name passed to the provider chat-completion API.
    topgenenumber : int, default=10
        Maximum number of marker genes retained for each cluster.
    base_url : str or None, default=None
        Custom API base URL. If ``None``, inferred from ``provider``.

    Returns
    -------
    dict or str
        Returns a mapping from cluster IDs to predicted cell types when API
        credentials are present. Returns prompt text when ``AGI_API_KEY`` is
        unavailable.

    Examples
    --------
    >>> markers = {"0": ["LYZ", "S100A8"], "1": ["CD3D", "IL7R"]}
    >>> res = gpt4celltype(markers, tissuename="Blood", provider="qwen")
    """
    input=input_data.copy()
    input_data=input
    del_k=[]
    for k in input_data:
        if len(input_data[k])==0:
            del_k.append(k)
    for k in del_k:
        del input[k]
    
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    if QWEN_API_KEY == "":
        print("Note: AGI API key not found: returning the prompt itself.")
        API_flag = False
    else:
        API_flag = True

    if isinstance(input, dict):
        input = {k: 'unknown' if not v else ','.join(v[:topgenenumber]) for k, v in input.items()}
    elif isinstance(input, pd.DataFrame):
        # Filter genes with positive log fold change and group by cluster, selecting top genes
        input = input[input['logfoldchanges'] > 0]
        input = input.groupby('cluster')['names'].apply(lambda x: ','.join(x.iloc[:topgenenumber]))
    else:
        raise ValueError("Input must be either a dictionary of lists or a pandas DataFrame.")

    
    if not API_flag:
        message = f'Identify cell types of {tissuename} cells in {speciename} using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([f'{k}: {v}' for k, v in input.items()])
        return message
    else:
        print("Note: AGI API key found: returning the cell type annotations.")
        
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
        }
        
        cutnum = int(np.ceil(len(input) / 30))
        if cutnum > 1:
            cid = np.digitize(range(1, len(input) + 1), bins=np.linspace(1, len(input), cutnum + 1))
        else:
            cid = np.ones(len(input), dtype=int)
        
        allres = {}
        from tqdm import tqdm
        for i in tqdm(range(1, cutnum + 1)):
            id_list = [j for j, x in enumerate(cid) if x == i]
            flag = False
            while not flag:
                messages = [{"role": "user", 
                             "content": f'Identify cell types of {tissuename} cells using the following markers separately for each row. Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types.\n' + '\n'.join([input[list(input.keys())[j]] for j in id_list if input[list(input.keys())[j]] != 'unknown'])}]
                
                params = {
                    "model": model,
                    "messages": messages
                }
                
                
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=params,
                    stream=False
                )
                
                res = response.json()
                
                if 'choices' in res and len(res['choices']) > 0:
                    res_content = res['choices'][0]['message']['content'].split('\n')
                    #print(res_content)
                    if len(res_content) == len(id_list):
                        flag = True
                        for idx, cell_type in zip(id_list, res_content):
                            key = list(input.keys())[idx]
                            allres[key] = 'unknown' if input[key] == 'unknown' else cell_type.strip(',')
        
        print('Note: It is always recommended to check the results returned by GPT-4 in case of AI hallucination, before going to downstream analysis.')
        for k in del_k:
            allres[k]='Unknown'
        return allres
    


def get_cluster_celltype(cluster_celltypes, cluster_markers, species, organization,
                        model,base_url,provider):
    """Resolve the best cell type per cluster from candidate labels and markers.

    Parameters
    ----------
    cluster_celltypes : dict
        Candidate cell types for each cluster,
        typically ``dict[cluster_id -> list[str]]``.
    cluster_markers : dict
        Marker genes per cluster, typically ``dict[cluster_id -> list[str]]``.
    species : str
        Species context used to constrain annotation.
    organization : str
        Tissue or organ context used in the prompt.
    model : str
        Chat model name used for API requests.
    base_url : str or None
        Base URL for chat-completion endpoint.
    provider : str
        Provider preset used when ``base_url`` is not supplied.

    Returns
    -------
    dict
        Mapping ``cluster_id -> predicted_cell_type``.
    """
    import os
    import numpy as np
    import pandas as pd
    import requests
    if base_url is None:
        if provider == 'openai':
            base_url = "https://api.openai.com/v1"
        elif provider == 'kimi':
            base_url = "https://api.moonshot.cn/v1"
        elif provider == 'qwen':
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY = os.getenv("AGI_API_KEY")
    
    
    # 在这里配置您在本站的API_KEY
    api_key = QWEN_API_KEY
    
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }
    cluster_celltype = {}
    from tqdm import tqdm
    for cluster_id, celltypes in tqdm(cluster_celltypes.items()):
        markers = cluster_markers.get(cluster_id, [])
        question = (
                    f"Given the species: {species} and organization: {organization}, "
                    f"determine the most suitable cell type for cluster {cluster_id}. "
                    f"The possible cell types are: {', '.join(celltypes)}. "
                    f"The gene markers for this cluster are: {', '.join(markers)}. "
                    f"Which cell type best represents this cluster? "
                    f"Only provide the cell type name. Do not show numbers before the name. Some can be a mixture of multiple cell types."
                    f"Do not provide the plural form of celltype."
                )
        #print(question)
        
        params = {
            "messages": [
        
                {
                    "role": 'user',
                    "content": question
                }
            ],
            # 如果需要切换模型，在这里修改
            "model": model
        }
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=params,
            stream=False
        )
        res = response.json()
        answer = res['choices'][0]['message']['content'].split('\n')
        # 将回答加入结果字典
        cluster_celltype[cluster_id] = answer[0].lower()
    
    return cluster_celltype
