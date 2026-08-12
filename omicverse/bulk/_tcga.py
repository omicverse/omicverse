
import os
import anndata
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from .._registry import register_function


_CLINICAL_COLUMN_ALIASES = {
    "case_submitter_id": (
        "case_submitter_id",
        "cases.submitter_id",
        "case.submitter_id",
        "submitter_id",
    ),
    "vital_status": (
        "vital_status",
        "demographic.vital_status",
        "cases.demographic.vital_status",
        "case.demographic.vital_status",
    ),
    "days_to_last_follow_up": (
        "days_to_last_follow_up",
        "diagnoses.days_to_last_follow_up",
        "cases.diagnoses.days_to_last_follow_up",
        "case.diagnoses.days_to_last_follow_up",
        "follow_ups.days_to_follow_up",
        "cases.follow_ups.days_to_follow_up",
        "case.follow_ups.days_to_follow_up",
    ),
    "days_to_death": (
        "days_to_death",
        "demographic.days_to_death",
        "cases.demographic.days_to_death",
        "case.demographic.days_to_death",
    ),
    "age_at_index": (
        "age_at_index",
        "demographic.age_at_index",
        "cases.demographic.age_at_index",
        "case.demographic.age_at_index",
    ),
    "tumor_grade": (
        "tumor_grade",
        "diagnoses.tumor_grade",
        "cases.diagnoses.tumor_grade",
        "case.diagnoses.tumor_grade",
    ),
}

_MISSING_CLINICAL_VALUES = {
    "",
    "--",
    "nan",
    "<na>",
    "na",
    "n/a",
    "none",
    "not reported",
    "not available",
    "unknown",
}


def _iter_reported_values(values):
    """Yield non-missing scalar values, including pipe-delimited GDC exports."""
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            parts = value
        else:
            parts = str(value).split("|")
        for part in parts:
            text = str(part).strip()
            if text.lower() not in _MISSING_CLINICAL_VALUES:
                yield part


def _first_reported(values, default=np.nan):
    return next(_iter_reported_values(values), default)


def _max_reported_days(values):
    days = pd.to_numeric(list(_iter_reported_values(values)), errors="coerce")
    days = np.asarray(days, dtype=float)
    days = days[np.isfinite(days)]
    return float(days.max()) if len(days) else np.nan


def _canonical_vital_status(values):
    statuses = [str(value).strip() for value in _iter_reported_values(values)]
    normalized = [status.lower() for status in statuses]
    if any(status in {"dead", "deceased"} for status in normalized):
        return "Dead"
    if "alive" in normalized:
        return "Alive"
    return statuses[0] if statuses else "Not Reported"


def _resolve_clinical_column(columns, canonical_name, explicit_name=None, required=False):
    available = list(columns)
    if explicit_name is not None:
        if explicit_name not in columns:
            raise KeyError(
                f"Clinical column {explicit_name!r}, mapped from {canonical_name!r}, "
                f"was not found. Available columns: {available}."
            )
        return explicit_name

    by_lower_name = {str(column).lower(): column for column in columns}
    for alias in _CLINICAL_COLUMN_ALIASES[canonical_name]:
        if alias.lower() in by_lower_name:
            return by_lower_name[alias.lower()]

    if required:
        raise KeyError(
            f"Required clinical field {canonical_name!r} was not found. Pass its "
            "actual column name with "
            f"clinical_columns={{'{canonical_name}': 'your_column'}}. "
            f"Available columns: {available}."
        )
    return None


@register_function(
    aliases=["TCGA分析", "pyTCGA", "tcga_analysis", "癌症基因组分析"],
    category="bulk",
    description="TCGA (The Cancer Genome Atlas) data analysis including survival analysis",
    examples=[
        "# Initialize TCGA analysis",
        "tcga = ov.bulk.pyTCGA(gdc_sample_sheet, gdc_download_files, clinical_cart)",
        "# Initialize AnnData object from TCGA data",
        "tcga.adata_init()",
        "# Or read existing AnnData",
        "tcga.adata_read('tcga_data.h5ad')",
        "# Initialize metadata",
        "tcga.adata_meta_init()",
        "# Initialize survival data",
        "tcga.survial_init()",
        "# Perform survival analysis for single gene",
        "tcga.survival_analysis('TP53', layer='deseq_normalize', plot=True)",
        "# Perform survival analysis for all genes",
        "tcga.survial_analysis_all()"
    ],
    related=["bulk.pyDEG", "utils.download_geneid_annotation_pair", "pl.survival_plot"]
)
class pyTCGA(object):
    r"""
    TCGA (The Cancer Genome Atlas) data analysis module.
    
    This class provides comprehensive functionality for downloading, processing,
    and analyzing TCGA genomic and clinical data.
    """
    def __init__(self,gdc_sample_sheep:str,gdc_download_files:str,clinical_cart:str):
        r"""Initialize TCGA analysis module.

        Arguments:
            gdc_sample_sheep: Path to TCGA Sample Sheet TSV file
            gdc_download_files: Path to downloaded TCGA data files directory
            clinical_cart: Path to TCGA clinical data tar.gz file

        """
        self.gdc_sample_sheep=gdc_sample_sheep
        self.gdc_download_files=gdc_download_files
        self.clinical_cart=clinical_cart
        exist_files=[i for i in os.listdir(gdc_download_files) if 'txt' not in i]
        
        self.sample_sheet=pd.read_csv(self.gdc_sample_sheep,sep='\t',index_col=0)
        exist_files=list(set(exist_files) & set(self.sample_sheet.index))
        self.sample_sheet=self.sample_sheet.loc[exist_files]
        self.clinical_sheet=pd.read_csv('{}/clinical.tsv'.format(self.clinical_cart),sep='\t',index_col=0)
        #self.clinical_sheet=self.clinical_sheet.loc[exist_files]
        

        sample_index=self.sample_sheet.index[0]
        sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
        sample_file_id=sample_index
        sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
        self.data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
        print('tcga module init success')
        
        
    def adata_read(self,path:str):
        r"""Read AnnData object from file.

        Arguments:
            path: Path to AnnData file
        """
        print('... anndata reading')
        import anndata as ad
        self.adata=ad.read_h5ad(path)
        
    def adata_init(self):
        self.index_init()
        self.expression_init()
        self.matrix_construct()
        
    def adata_meta_init(self,var_names:list=['gene_name','gene_type'],
                  obs_names:list=['Case ID','Sample Type'])->anndata.AnnData:
        r"""Initialize AnnData metadata.

        Arguments:
            var_names: Column names for variable (gene) metadata (default: ['gene_name','gene_type'])
            obs_names: Column names for observation (sample) metadata (default: ['Case ID','Sample Type'])

        Returns:
            adata: AnnData object with initialized metadata

        """
        print('...anndata meta init',var_names,obs_names)
        adata=self.adata
        #var_pd=pd.DataFrame(index=self.adata.var.index)
        var_pd=self.data_test.loc[adata.var.index,var_names]
        var_pd['gene_id']=var_pd.index.tolist()
        var_pd.index=var_pd['gene_name'].values
        #obs_pd=pd.DataFrame(index=data_pd_count.columns)
        sample_sheet_tmp=self.sample_sheet.copy()
        sample_sheet_tmp.index=sample_sheet_tmp['Sample ID']
        obs_pd=sample_sheet_tmp.loc[adata.obs.index,obs_names]
        obs_pd=obs_pd[~obs_pd.index.duplicated(keep='first')]
        adata.obs=obs_pd.loc[adata.obs.index]
        adata.var=var_pd
        adata.var.index=adata.var['gene_name'].astype('str').values
        adata.var_names_make_unique()
        self.adata=adata
        return adata
        
    def survial_init(
        self,
        clinical_columns: Optional[Dict[str, str]] = None,
        obs_case_id: str = "Case ID",
    ) -> None:
        r"""Initialize survival analysis data.

        Processes clinical data to extract survival information including
        vital status and survival days. Both legacy unprefixed columns and
        current GDC fields such as ``demographic.vital_status`` and
        ``diagnoses.days_to_last_follow_up`` are detected automatically.

        Arguments:
            clinical_columns: Optional mapping from canonical field names to
                columns in ``clinical_sheet``. Supported canonical names are
                ``case_submitter_id``, ``vital_status``,
                ``days_to_last_follow_up``, ``days_to_death``,
                ``age_at_index``, and ``tumor_grade``.
            obs_case_id: Column in ``adata.obs`` containing GDC case submitter
                identifiers (default: ``'Case ID'``).

        Updates ``self.adata`` in place, restricting it to samples with
        matching clinical cases and annotating ``vital_status`` and ``days``.
        """
        clinical_columns = dict(clinical_columns or {})
        unknown_fields = sorted(
            set(clinical_columns).difference(_CLINICAL_COLUMN_ALIASES)
        )
        if unknown_fields:
            raise ValueError(
                "Unknown canonical clinical field(s): "
                f"{unknown_fields}. Supported fields are: "
                f"{sorted(_CLINICAL_COLUMN_ALIASES)}."
            )
        if obs_case_id not in self.adata.obs:
            raise KeyError(
                f"adata.obs has no case identifier column {obs_case_id!r}."
            )

        source = self.clinical_sheet
        resolved = {
            field: _resolve_clinical_column(
                source.columns,
                field,
                clinical_columns.get(field),
                required=field in {"case_submitter_id", "vital_status"},
            )
            for field in _CLINICAL_COLUMN_ALIASES
        }
        if (
            resolved["days_to_last_follow_up"] is None
            and resolved["days_to_death"] is None
        ):
            raise KeyError(
                "No survival-time field was found. Provide "
                "'days_to_last_follow_up' and/or 'days_to_death' through "
                "clinical_columns."
            )

        clinical = pd.DataFrame(index=source.index)
        for field, column in resolved.items():
            clinical[field] = source[column] if column is not None else np.nan
        clinical["case_submitter_id"] = [
            str(_first_reported([value], default="")).strip()
            for value in clinical["case_submitter_id"]
        ]
        clinical = clinical.loc[clinical["case_submitter_id"] != ""]

        records = []
        for case_id, case_rows in clinical.groupby(
            "case_submitter_id", sort=False
        ):
            vital_status = _canonical_vital_status(case_rows["vital_status"])
            follow_up = _max_reported_days(
                case_rows["days_to_last_follow_up"]
            )
            death = _max_reported_days(case_rows["days_to_death"])
            if vital_status == "Dead":
                days = death if np.isfinite(death) else follow_up
            elif vital_status == "Alive":
                days = follow_up if np.isfinite(follow_up) else death
            else:
                finite_days = [day for day in (follow_up, death) if np.isfinite(day)]
                days = max(finite_days) if finite_days else np.nan
            records.append(
                {
                    "case_submitter_id": case_id,
                    "vital_status": vital_status,
                    "days_to_last_follow_up": follow_up,
                    "days_to_death": death,
                    "age_at_index": _first_reported(case_rows["age_at_index"]),
                    "tumor_grade": _first_reported(case_rows["tumor_grade"]),
                    "days": days,
                }
            )

        if not records:
            raise ValueError(
                "No valid case submitter identifiers were found in the clinical data."
            )
        s_pd = pd.DataFrame.from_records(records).set_index("case_submitter_id")
        self.s_pd = s_pd

        case_ids = self.adata.obs[obs_case_id].astype(str)
        matched = case_ids.isin(s_pd.index)
        self.adata = self.adata[matched].copy()
        case_ids = self.adata.obs[obs_case_id].astype(str)
        self.adata.obs["vital_status"] = (
            case_ids.map(s_pd["vital_status"]).fillna("Not Reported")
        )
        self.adata.obs["days"] = pd.to_numeric(
            case_ids.map(s_pd["days"]), errors="coerce"
        )
        
        
        
    def index_init(self)->list:
        r"""Initialize gene indices for AnnData construction.

        Returns:
            all_lncRNA_index: List of all gene indices from TCGA samples
        """
        print('...index init')
        all_lncRNA_index=[]
        for sample_index in self.sample_sheet.index:
            sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
            sample_file_id=sample_index
            sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
            data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
            #data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s=data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            all_lncRNA_index=list(set(all_lncRNA_index) | set(data_c_s.index.tolist()))
        self.tcga_index=all_lncRNA_index
        return all_lncRNA_index
    
    def expression_init(self):
        r"""Initialize expression matrices for TCGA data.
        
        Creates count, TPM, and FPKM expression matrices from TCGA files.
        """
        print('... expression matrix init')
        data_pd_count=pd.DataFrame(index=self.tcga_index)
        data_pd_tpm=pd.DataFrame(index=self.tcga_index)
        data_pd_fpkm=pd.DataFrame(index=self.tcga_index)

        for sample_index in self.sample_sheet.index:
            sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
            sample_file_id=sample_index
            sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
            #print(sample_id)
            data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
            #data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s=data_test['unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_count[sample_id]=0
            data_pd_count.loc[data_c_s.index,sample_id]=data_c_s.values

            data_c_s=data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_tpm[sample_id]=0
            data_pd_tpm.loc[data_c_s.index,sample_id]=data_c_s.values

            data_c_s=data_test['fpkm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_fpkm[sample_id]=0
            data_pd_fpkm.loc[data_c_s.index,sample_id]=data_c_s.values
            
        self.data_pd_count=data_pd_count
        self.data_pd_tpm=data_pd_tpm
        self.data_pd_fpkm=data_pd_fpkm
        self.data_test=data_test
    
    def matrix_construct(self):
        r"""Construct AnnData object from expression matrices.
        
        Creates AnnData object with multiple layers including raw counts,
        TPM, FPKM, and DESeq2-normalized expression.
        """
        print('...anndata construct')
        var_pd=pd.DataFrame(index=self.data_pd_count.index)
        obs_pd=pd.DataFrame(index=self.data_pd_count.columns)
        adata=anndata.AnnData(self.data_pd_count.T,var=var_pd,obs=obs_pd)
        adata.layers['tpm']=self.data_pd_tpm.T.values
        adata.layers['fpkm']=self.data_pd_fpkm.T.values
        adata.layers['deseq_normalize']=self.matrix_normalize(self.data_pd_count).T.values
        self.adata=adata
        return adata
    
    def matrix_normalize(self,data:pd.DataFrame)->pd.DataFrame:
        r"""Normalize expression matrix using DESeq2 method.

        Arguments:
            data: Raw count expression matrix to normalize

        Returns:
            data: DESeq2-normalized expression matrix
        """
        avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        data1=data.loc[avg1.index]
        data_log=data1.apply(np.log,axis=1)
        scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
        return data/scale

    
    
    def survival_analysis(self,gene:str,layer:str='raw',plot:bool=False,gene_threshold:str='median')->Tuple[float,float]:
        r"""Perform survival analysis for a specific gene.

        Arguments:
            gene: Gene name for survival analysis
            layer: AnnData layer to use for expression values (default: 'raw')
            plot: Whether to generate Kaplan-Meier survival plot (default: False)
            gene_threshold: Method to split samples into high/low expression groups
                          (default: 'median', options: 'median', 'mean', or numeric value)

        Returns:
            test_statistic: Log-rank test statistic
            pvalue: Log-rank test p-value

        """
        from scipy.sparse import issparse
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        goal_gene=gene
        
        s_pd=self.s_pd
        s_pd=s_pd.loc[self.adata.obs['Case ID']]
        if layer!='raw':
            if layer not in self.adata.layers.keys():
                #issparse
                
                if issparse(self.adata.X):
                    s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
                else:
                    s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1)
            else:

                if issparse(self.adata.layers[layer]):
                    s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].layers[layer].mean(axis=1).toarray()
                else:
                    s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].layers[layer].mean(axis=1)
            
        else:
            if issparse(self.adata.X):
                s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
            else:
                s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1)
        if gene_threshold=='median':
            s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].median() else 'Low' for i in s_pd[goal_gene] ]
        elif gene_threshold=='mean':
            s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].mean() else 'Low' for i in s_pd[goal_gene] ]
        else:
            s_pd['{}_status'.format(goal_gene)]=['High' if i>gene_threshold else 'Low' for i in s_pd[goal_gene] ]
        s_pd=s_pd.loc[s_pd['days']!="'--"]
        s_pd['fustat'] = [0 if 'Alive'==i else 1 for i in s_pd['vital_status']]
        s_pd['gene_fustat'] = [0 if 'High'==i else 1 for i in s_pd['{}_status'.format(goal_gene)]]

        km = KaplanMeierFitter()
        T = s_pd['days'].astype(float) / 365
        E=s_pd['fustat']

        gender = (s_pd['{}_status'.format(goal_gene)] == 'High')
        lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
        if plot==True:
            fig, ax = plt.subplots(figsize=(3,3))
            km.fit(T[gender], event_observed=E[gender], label="High")
            km.plot(ax=ax,color='#941456')
            km.fit(T[~gender], event_observed=E[~gender], label="Low")
            km.plot(ax=ax,color='#368650')
            lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
            lr.p_value

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            plt.xlabel('Years')
            plt.ylabel('Pecent Survial')
            plt.title('Survial: {}\np-value: {}'.format(goal_gene,round(lr.p_value,3)))
            plt.grid(False)
            
        return lr.test_statistic,lr.p_value
    
    def survial_analysis_all(self):
        r"""Perform survival analysis for all genes in the dataset.
        
        Calculates survival statistics for every gene and stores results
        in AnnData.var as 'survial_test_statistic' and 'survial_p' columns.
        """
        from tqdm import tqdm
        res_l_lnc=[]
        res_l_tt=[]
        for i in tqdm(self.adata.var.index):
            res_l_tt.append(self.survival_analysis(i)[0])
            res_l_lnc.append(self.survival_analysis(i)[1])
        self.adata.var['survial_test_statistic']=res_l_tt
        self.adata.var['survial_p']=res_l_lnc
