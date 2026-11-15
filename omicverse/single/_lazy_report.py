import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
import os
from .._registry import register_function
warnings.filterwarnings('ignore')

from ..pl import *


class HTMLReportGenerator:
    """HTML报告生成器，使用模板系统"""
    
    def __init__(self, template_dir=None):
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.template_dir = template_dir
        
        # Load templates
        self.load_templates()
    
    def load_templates(self):
        """加载HTML模板"""
        try:
            with open(os.path.join(self.template_dir, 'report_template.html'), 'r', encoding='utf-8') as f:
                self.main_template = f.read()
            
            with open(os.path.join(self.template_dir, 'sections.html'), 'r', encoding='utf-8') as f:
                sections_content = f.read()
                
            # Extract individual section templates
            self.section_templates = {}
            import re
            
            # Extract templates using regex
            templates = re.findall(r'<template id="([^"]+)">(.*?)</template>', sections_content, re.DOTALL)
            for template_id, template_content in templates:
                self.section_templates[template_id] = template_content.strip()
                
        except FileNotFoundError as e:
            print(f"Warning: Template file not found: {e}")
            # Fallback to embedded templates if files don't exist
            self.main_template = self._get_embedded_main_template()
            self.section_templates = self._get_embedded_section_templates()
    
    def _get_embedded_main_template(self):
        """备用的嵌入式主模板"""
        return """<!DOCTYPE html>
<html lang="en" class="day-mode">
<head>
    <meta charset="UTF-8">
    <title>scRNA-seq Analysis Report</title>
    <style>{{embedded_css}}</style>
</head>
<body>
    {{body_content}}
    <script>{{embedded_js}}</script>
</body>
</html>"""
    
    def _get_embedded_section_templates(self):
        """备用的嵌入式部分模板"""
        return {
            'qc-section-template': '''
                <div id="quality" class="card fade-in">
                    <div class="card-header">
                        <h2 class="card-title"><i class="card-icon">🔍</i> Quality Control</h2>
                    </div>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{{qc_plot_day}}" class="plot-img day-mode">
                        <img src="data:image/png;base64,{{qc_plot_night}}" class="plot-img night-mode" style="display:none;">
                    </div>
                </div>
            ''',
            # Add other templates as needed...
        }
    
    def render_section(self, section_name, **kwargs):
        """渲染特定部分"""
        template_key = f"{section_name}-section-template"
        if template_key in self.section_templates:
            template = self.section_templates[template_key]
            # Simple template substitution
            for key, value in kwargs.items():
                template = template.replace(f'{{{{{key}}}}}', str(value))
            return template
        return ""
    
    def render_main(self, **kwargs):
        """渲染主模板"""
        template = self.main_template
        
        # 添加大学logo HTML
        kwargs['university_logos_html'] = self._get_university_logos_html()
        
        for key, value in kwargs.items():
            template = template.replace(f'{{{{{key}}}}}', str(value))
        return template

    def _get_university_logos_html(self):
        """生成大学logo的HTML，使用fig_to_base64函数"""
        img_dir = os.path.join(self.template_dir, 'img')
        
        universities = [
            {
                'name': 'Stanford<br>University',
                'filename': 'stanford-logo.png',
                'alt': 'Stanford University',
                'fallback_emoji': '🏛️'
            },
            {
                'name': 'Sun Yat-sen<br>University', 
                'filename': 'sun-yet-logo.png',
                'alt': 'Sun Yat-sen University',
                'fallback_emoji': '🏛️'
            },
            {
                'name': 'Beijing University of<br>Science and Technology',
                'filename': 'ustb-logo.png', 
                'alt': 'University of Science and Technology Beijing',
                'fallback_emoji': '🏛️'
            }
        ]
        
        def logo_to_base64(img_path):
            """读取logo并转换为base64，优先保持原始透明背景"""
            try:
                # 方法1：直接读取文件（最佳，完全保持原始透明度）
                with open(img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_str = base64.b64encode(img_data).decode()
                    return img_str
                    
            except Exception as e:
                print(f"Warning: Direct file read failed for {img_path}: {e}")
                
                # 方法2：matplotlib处理（保持透明背景的fallback）
                try:
                    import matplotlib.pyplot as plt
                    import matplotlib.image as mpimg
                    
                    # 读取图片
                    img = mpimg.imread(img_path)
                    
                    # 创建figure，使用透明背景
                    fig, ax = plt.subplots(figsize=(2, 2), facecolor='none')
                    ax.imshow(img)
                    ax.axis('off')  # 隐藏坐标轴
                    plt.tight_layout(pad=0)
                    
                    # 保存为PNG格式，保持透明背景
                    buffer = BytesIO()
                    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                               facecolor='none', edgecolor='none', transparent=True)
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    plt.close(fig)
                    
                    return img_str
                    
                except Exception as e2:
                    print(f"Warning: All methods failed to convert {img_path} to base64: {e2}")
                    return None
        
        logo_items = []
        for uni in universities:
            img_path = os.path.join(img_dir, uni['filename'])
            
            if os.path.exists(img_path):
                # 图片存在，转换为base64
                base64_img = logo_to_base64(img_path)
                if base64_img:
                    logo_html = f'''<img src="data:image/png;base64,{base64_img}" alt="{uni['alt']}" class="footer-logo">'''
                else:
                    # base64转换失败，使用emoji回退
                    logo_html = f'''<span class="footer-logo-fallback">{uni['fallback_emoji']}</span>'''
            else:
                # 图片不存在，使用emoji作为回退
                logo_html = f'''<span class="footer-logo-fallback">{uni['fallback_emoji']}</span>'''
            logo_items.append(logo_html)
        
        return ''.join(logo_items)


@register_function(
    aliases=['单细胞分析报告', 'generate_scRNA_report', 'scRNA report'],
    category="single",
    description="Generate an HTML report summarizing single-cell QC, clustering, markers, and embedding views for reproducible interpretation.",
    prerequisites={'optional_functions': ['lazy', 'pp.preprocess', 'pp.neighbors', 'pp.umap']},
    requires={'obs': ['cluster labels'], 'obsm': ['X_umap (recommended)']},
    produces={'uns': ['analysis_report']},
    auto_fix='none',
    examples=['ov.single.generate_scRNA_report(adata, output_path="reports/pbmc_report.html", species="human")'],
    related=['single.lazy', 'utils.plot_embedding_celltype']
)
def generate_scRNA_report(adata, output_path="scRNA_analysis_report.html", 
                         species='human', sample_key=None, template_dir=None,
                         enable_analytics=True, analytics_id="OV-001"):
    """
    Generate a MultiQC-style HTML report for single-cell RNA-seq analysis.

    Parameters
    ----------
    adata:AnnData object
        Analyzed single-cell AnnData, typically produced by ``ov.single.lazy``.
    output_path:str
        Output path of generated HTML report.
    species:str
        Species label shown in report metadata.
    sample_key:str or None
        Column in ``adata.obs`` used as sample/batch grouping.
    template_dir:str or None
        Directory containing HTML templates. If ``None``, built-in templates are used.
    enable_analytics:bool
        Whether analytics snippet is injected into final HTML.
    analytics_id:str
        Analytics identifier used when tracking is enabled.

    Returns
    -------
    str
        Path to saved report file.
    """
    
    # Initialize report generator
    generator = HTMLReportGenerator(template_dir)
    
    # Set scanpy settings for clean plots
    plt.style.use('default')

    if sample_key is None:
        sample_key = 'batch_none'
        adata.obs[sample_key] = 'sample1'

    # Color schemes
    color_schemes = {
        "day": {
            "primary": "#1f6912",
            "plot_bg": "white"
        },
        "night": {
            "primary": "#1c9519",
            "plot_bg": "#1e293b"
        }
    }

    # 修改图像生成函数支持夜间模式
    def fig_to_base64(fig, mode="day"):
        """Convert matplotlib figure to base64 string with theme support"""
        facecolor = color_schemes[mode]["plot_bg"]
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor=facecolor, edgecolor='none')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_str
    
    # Extract analysis status and parameters
    status = adata.uns.get('status', {})
    status_args = adata.uns.get('status_args', {})
    
    # Basic data information
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # Calculate additional statistics
    median_genes = np.median(adata.obs['n_genes_by_counts']) if 'n_genes_by_counts' in adata.obs else 0
    median_umis = np.median(adata.obs['total_counts']) if 'total_counts' in adata.obs else 0
    n_hvgs = sum(adata.var['highly_variable']) if 'highly_variable' in adata.var else 0
    n_clusters = len(adata.obs['best_clusters'].unique()) if 'best_clusters' in adata.obs else 0
    
    # 生成两种模式的图像
    plots = {"day": {}, "night": {}}
    
    for mode in ["day", "night"]:
        # 设置绘图主题
        if mode == "night":
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # 1. QC metrics violin plot
        if all(col in adata.obs for col in ['detected_genes', 'nUMIs', 'mito_perc']):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), 
                                    facecolor=color_schemes[mode]["plot_bg"])
            for i,key in enumerate(['detected_genes', 'nUMIs', 'mito_perc']):
                half_violin_boxplot(
                    adata,
                    key,
                    groupby=sample_key,
                    ax=axes[i],
                )
            
            plt.suptitle('Quality Control Metrics Distribution', fontsize=16, y=1.02)
            plots[mode]['qc_violin'] = fig_to_base64(fig, mode)
        
        # 2. PCA plot
        if 'X_pca' in adata.obsm:
            fig = plt.figure(figsize=(12, 5), facecolor=color_schemes[mode]["plot_bg"])
            
            # PCA variance ratio
            plt.subplot(1, 2, 1)
            from ..pl._plot_backend import plot_pca_variance_ratio1
            plot_pca_variance_ratio1(adata)
            plt.title('PCA Variance Explained', fontsize=12)
            
            # PCA scatter plot
            plt.subplot(1, 2, 2)
            if sample_key and sample_key in adata.obs:
                embedding(adata, basis='X_pca', color=sample_key, ax=plt.gca(), show=False, frameon=False)
            else:
                embedding(adata, basis='X_pca', ax=plt.gca(), show=False, frameon=False)
            plt.title('PCA Visualization', fontsize=12)
            
            plt.tight_layout()
            plots[mode]['pca_plot'] = fig_to_base64(fig, mode)
        
        # 3. Batch correction comparison with UMAP
        if 'X_umap_harmony' in adata.obsm and 'X_umap_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=color_schemes[mode]["plot_bg"])
            
            # Original PCA
            embedding(adata, basis='X_pca', color=sample_key, ax=axes[0,0], show=False, frameon=False, title='Original PCA')
            
            # Harmony UMAP
            embedding(adata, basis='X_umap_harmony', color=sample_key, ax=axes[0,1], 
                           show=False, frameon='small', title='Harmony UMAP')
            
            # scVI UMAP
            embedding(adata, basis='X_umap_scVI', color=sample_key, ax=axes[0,2], 
                           show=False, frameon='small', title='scVI UMAP')
            
            # Cell cycle visualization if available
            if 'phase' in adata.obs:
                embedding(adata, basis='X_pca', color='phase', ax=axes[1,0], show=False, frameon=False, title='Cell Cycle - PCA')
                embedding(adata, basis='X_umap_harmony', color='phase', ax=axes[1,1], 
                               show=False, frameon='small', title='Cell Cycle - Harmony UMAP')
                embedding(adata, basis='X_umap_scVI', color='phase', ax=axes[1,2], 
                               show=False, frameon='small', title='Cell Cycle - scVI UMAP')
            
            plt.tight_layout()
            plots[mode]['batch_correction'] = fig_to_base64(fig, mode)
        
        # Fallback: if only X_harmony and X_scVI are available (backward compatibility)
        elif 'X_harmony' in adata.obsm and 'X_scVI' in adata.obsm and sample_key and sample_key in adata.obs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=color_schemes[mode]["plot_bg"])
            
            # Original PCA
            embedding(adata, basis='X_pca', color=sample_key, ax=axes[0,0], show=False, frameon=False, title='Original PCA')
            
            # Harmony
            embedding(adata, basis='X_harmony', color=sample_key, ax=axes[0,1], 
                           show=False, frameon='small', title='Harmony Integration')
            
            # scVI
            embedding(adata, basis='X_scVI', color=sample_key, ax=axes[0,2], 
                           show=False, frameon='small', title='scVI Integration')
            
            # Cell cycle visualization if available
            if 'phase' in adata.obs:
                embedding(adata, basis='X_pca', color='phase', ax=axes[1,0], show=False, frameon=False, title='Cell Cycle - PCA')
                embedding(adata, basis='X_harmony', color='phase', ax=axes[1,1], 
                               show=False, frameon='small', title='Cell Cycle - Harmony')
                embedding(adata, basis='X_scVI', color='phase', ax=axes[1,2], 
                               show=False, frameon='small', title='Cell Cycle - scVI')
            
            plt.tight_layout()
            plots[mode]['batch_correction'] = fig_to_base64(fig, mode)
        
        # 4. Clustering visualization with multiple methods
        if 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
            # Create a larger figure for multiple clustering results
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor=color_schemes[mode]["plot_bg"])
            
            # Best clusters (MDE)
            embedding(adata, basis='X_mde', color='best_clusters', ax=axes[0,0],
                           show=False, frameon='small', title='Best Clusters (SCCAF)')
            
            # Additional clustering methods if available
            if 'leiden_clusters_L1' in adata.obs:
                embedding(adata, basis='X_mde', color='leiden_clusters_L1', ax=axes[0,1],
                               show=False, frameon='small', title='Leiden Clusters (L1)')
            else:
                axes[0,1].text(0.5, 0.5, 'Leiden L1\nNot Available', ha='center', va='center', transform=axes[0,1].transAxes)
                axes[0,1].set_xticks([])
                axes[0,1].set_yticks([])
            
            if 'louvain_clusters_L1' in adata.obs:
                embedding(adata, basis='X_mde', color='louvain_clusters_L1', ax=axes[0,2],
                               show=False, frameon='small', title='Louvain Clusters (L1)')
            else:
                axes[0,2].text(0.5, 0.5, 'Louvain L1\nNot Available', ha='center', va='center', transform=axes[0,2].transAxes)
                axes[0,2].set_xticks([])
                axes[0,2].set_yticks([])
            
            # Sample distribution if available
            if sample_key and sample_key in adata.obs:
                embedding(adata, basis='X_mde', color=sample_key, ax=axes[1,0],
                               show=False, frameon='small', title='Sample Distribution')
            else:
                # Alternative: show total UMI counts
                if 'total_counts' in adata.obs:
                    embedding(adata, basis='X_mde', color='total_counts', ax=axes[1,0],
                                   show=False, frameon='small', title='Total UMI Counts')
                else:
                    axes[1,0].text(0.5, 0.5, 'Sample Info\nNot Available', ha='center', va='center', transform=axes[1,0].transAxes)
                    axes[1,0].set_xticks([])
                    axes[1,0].set_yticks([])
            
            # Additional clustering methods (L2)
            if 'leiden_clusters_L2' in adata.obs:
                embedding(adata, basis='X_mde', color='leiden_clusters_L2', ax=axes[1,1],
                               show=False, frameon='small', title='Leiden Clusters (L2)')
            else:
                axes[1,1].text(0.5, 0.5, 'Leiden L2\nNot Available', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_xticks([])
                axes[1,1].set_yticks([])
            
            if 'louvain_clusters_L2' in adata.obs:
                embedding(adata, basis='X_mde', color='louvain_clusters_L2', ax=axes[1,2],
                               show=False, frameon='small', title='Louvain Clusters (L2)')
            else:
                axes[1,2].text(0.5, 0.5, 'Louvain L2\nNot Available', ha='center', va='center', transform=axes[1,2].transAxes)
                axes[1,2].set_xticks([])
                axes[1,2].set_yticks([])
            
            plt.tight_layout()
            plots[mode]['clustering'] = fig_to_base64(fig, mode)
        
        # Fallback for simpler clustering visualization
        elif 'best_clusters' in adata.obs and 'X_mde' in adata.obsm:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=color_schemes[mode]["plot_bg"])
            
            # Clusters
            embedding(adata, basis='X_mde', color='best_clusters', ax=axes[0],
                           show=False, frameon='small', title='Final Clusters')
            
            # Sample distribution if available
            if sample_key and sample_key in adata.obs:
                embedding(adata, basis='X_mde', color=sample_key, ax=axes[1],
                               show=False, frameon='small', title='Sample Distribution')
            else:
                # Alternative: show total UMI counts
                if 'total_counts' in adata.obs:
                    embedding(adata, basis='X_mde', color='total_counts', ax=axes[1],
                                   show=False, frameon='small', title='Total UMI Counts')
            
            plt.tight_layout()
            plots[mode]['clustering'] = fig_to_base64(fig, mode)
        
        # 5. Cell cycle distribution
        if 'phase' in adata.obs:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=color_schemes[mode]["plot_bg"])
            
            # Phase distribution
            phase_counts = adata.obs['phase'].value_counts()
            axes[0].pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%',
                       colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0].set_title('Cell Cycle Phase Distribution', fontsize=14)
            
            # Phase by sample if available
            if sample_key and sample_key in adata.obs:
                phase_sample = pd.crosstab(adata.obs[sample_key], adata.obs['phase'], normalize='index') * 100
                phase_sample.plot(kind='bar', ax=axes[1], rot=45)
                axes[1].set_title('Cell Cycle Phase by Sample (%)', fontsize=14)
                axes[1].legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Alternative: show cell cycle scores
                if all(col in adata.obs for col in ['S_score', 'G2M_score']):
                    axes[1].scatter(adata.obs['S_score'], adata.obs['G2M_score'], 
                                  c=adata.obs['phase'].astype('category').cat.codes, 
                                  alpha=0.6, s=10)
                    axes[1].set_xlabel('S Score')
                    axes[1].set_ylabel('G2M Score')
                    axes[1].set_title('Cell Cycle Scores', fontsize=14)
            
            plt.tight_layout()
            plots[mode]['cell_cycle'] = fig_to_base64(fig, mode)
        
        # 6. QC metrics correlation heatmap
        qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        if 'doublet_score' in adata.obs:
            qc_cols.append('doublet_score')
        if 'S_score' in adata.obs:
            qc_cols.extend(['S_score', 'G2M_score'])
        
        available_qc_cols = [col for col in qc_cols if col in adata.obs]
        if len(available_qc_cols) > 2:
            fig = plt.figure(figsize=(8, 6), facecolor=color_schemes[mode]["plot_bg"])
            corr_matrix = adata.obs[available_qc_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            plt.title('QC Metrics Correlation Matrix', fontsize=14)
            plt.tight_layout()
            plots[mode]['qc_correlation'] = fig_to_base64(fig, mode)
        
        # 7. Gene expression overview
        if 'highly_variable' in adata.var:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=color_schemes[mode]["plot_bg"])
            
            # Mean vs variance of genes
            axes[0].scatter(adata.var['means'], adata.var['variances'], 
                           c=adata.var['highly_variable'], alpha=0.6, s=10)
            axes[0].set_xlabel('Mean Expression')
            axes[0].set_ylabel('Variance')
            axes[0].set_title('Gene Expression: Mean vs Variance')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            
            # Distribution of highly variable genes
            hvg_stats = adata.var.groupby('highly_variable').size()
            axes[1].bar(['Non-HVG', 'HVG'], hvg_stats.values, 
                       color=['#E0E0E0', '#4ECDC4'], alpha=0.8)
            axes[1].set_ylabel('Number of Genes')
            axes[1].set_title('Highly Variable Genes Selection')
            
            # Add percentage labels
            total_genes = sum(hvg_stats.values)
            for i, v in enumerate(hvg_stats.values):
                axes[1].text(i, v + total_genes*0.01, f'{v}\n({v/total_genes*100:.1f}%)', 
                            ha='center', va='bottom')
            
            plt.tight_layout()
            plots[mode]['gene_expression'] = fig_to_base64(fig, mode)
    
    # Prepare template data
    template_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'species': species.title(),
        'n_cells': f"{n_cells:,}",
        'n_genes': f"{n_genes:,}",
        'n_hvgs': f"{n_hvgs:,}",
        'median_genes': f"{int(median_genes):,}",
        'median_umis': f"{int(median_umis):,}",
        'n_clusters': str(n_clusters),
        'hvg_percentage': f"{n_hvgs/n_genes*100:.1f}",
        'progress_percentage': f"{len([k for k, v in status.items() if v])/len(status)*100:.0f}",
        'completed_steps': str(len([k for k, v in status.items() if v])),
        'total_steps': str(len(status)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'analytics_id': analytics_id
    }
    
    # Generate sections
    sections_html = {}
    
    # QC Section
    if 'qc_violin' in plots['day']:
        sections_html['qc_section'] = generator.render_section('qc', 
            qc_plot_day=plots['day']['qc_violin'],
            qc_plot_night=plots['night']['qc_violin'],
            mito_threshold=status_args.get('qc', {}).get('mito_perc', 'N/A'),
            min_umis=status_args.get('qc', {}).get('nUMIs', 'N/A'),
            min_genes=status_args.get('qc', {}).get('detected_genes', 'N/A')
        )
    else:
        sections_html['qc_section'] = ""
    
    # PCA Section
    if 'pca_plot' in plots['day']:
        sections_html['pca_section'] = generator.render_section('pca',
            pca_plot_day=plots['day']['pca_plot'],
            pca_plot_night=plots['night']['pca_plot'],
            n_pcs=status_args.get('pca', {}).get('n_pcs', 'N/A'),
            data_layer=status_args.get('pca', {}).get('layer', 'X (default)'),
            use_hvgs=status_args.get('pca', {}).get('use_highly_variable', 'True')
        )
    else:
        sections_html['pca_section'] = ""
    
    # Gene Expression Section
    if 'gene_expression' in plots['day']:
        sections_html['expression_section'] = generator.render_section('expression',
            expression_plot_day=plots['day']['gene_expression'],
            expression_plot_night=plots['night']['gene_expression'],
            n_hvgs=n_hvgs,
            n_genes=n_genes,
            hvg_percentage=f"{n_hvgs/n_genes*100:.1f}"
        )
    else:
        sections_html['expression_section'] = ""
    
    # Batch Correction Section
    if 'batch_correction' in plots['day']:
        sections_html['batch_section'] = generator.render_section('batch',
            batch_plot_day=plots['day']['batch_correction'],
            batch_plot_night=plots['night']['batch_correction'],
            best_method=adata.uns.get('bench_best_res', 'Unknown')
        )
    else:
        sections_html['batch_section'] = ""
    
    # Clustering Section
    if 'clustering' in plots['day']:
        sections_html['clustering_section'] = generator.render_section('clustering',
            clustering_plot_day=plots['day']['clustering'],
            clustering_plot_night=plots['night']['clustering'],
            n_clusters=n_clusters
        )
    else:
        sections_html['clustering_section'] = ""
    
    # Cell Cycle Section
    if 'cell_cycle' in plots['day']:
        sections_html['cellcycle_section'] = generator.render_section('cellcycle',
            cellcycle_plot_day=plots['day']['cell_cycle'],
            cellcycle_plot_night=plots['night']['cell_cycle']
        )
    else:
        sections_html['cellcycle_section'] = ""
    
    # Benchmark Section
    if 'bench_res' in adata.uns:
        sections_html['benchmark_section'] = generator.render_section('benchmark',
            bench_res=adata.uns['bench_res'],
            best_method=adata.uns['bench_best_res']
        )
    else:
        sections_html['benchmark_section'] = ""
    
    # Pipeline steps
    steps = [
        ('qc', 'Quality Control', '🔍'),
        ('preprocess', 'Preprocessing', '⚙️'),
        ('scaled', 'Data Scaling', '📏'),
        ('pca', 'PCA', '📈'),
        ('cell_cycle', 'Cell Cycle', '🔄'),
        ('harmony', 'Harmony', '🎵'),
        ('scVI', 'scVI', '🧬'),
        ('eval_bench', 'Benchmarking', '📊'),
        ('eval_clusters', 'Clustering', '🎯')
    ]
    
    pipeline_rows = []
    for step_key, step_name, emoji in steps:
        step_status = status.get(step_key, False)
        status_class = "status-badge status-success" if step_status else "status-badge status-danger"
        status_text = "Completed" if step_status else "Not Completed"
        
        step_params = status_args.get(step_key, {})
        params_text = ""
        if step_params:
            params_list = [f"{k}: {v}" for k, v in list(step_params.items())[:2]]
            params_text = "; ".join(params_list)
            if len(step_params) > 2:
                params_text += f" (+{len(step_params)-2})"
        else:
            params_text = "Default"
        
        pipeline_rows.append(f'''
            <tr>
                <td>{emoji} {step_name}</td>
                <td><span class="{status_class}">{status_text}</span></td>
                <td>{params_text}</td>
            </tr>
        ''')
    
    sections_html['pipeline_steps'] = ''.join(pipeline_rows)
    
    # Combine all template data
    template_data.update(sections_html)
    
    # Load CSS and JS if using embedded mode
    if not os.path.exists(os.path.join(generator.template_dir, 'report_template.html')):
        # Load embedded CSS and JS
        css_path = os.path.join(generator.template_dir, 'styles.css')
        js_path = os.path.join(generator.template_dir, 'script.js')
        
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                template_data['embedded_css'] = f.read()
        except:
            template_data['embedded_css'] = ""
            
        try:
            with open(js_path, 'r', encoding='utf-8') as f:
                template_data['embedded_js'] = f.read()
        except:
            template_data['embedded_js'] = ""
    
    # Generate final HTML
    html_content = generator.render_main(**template_data)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ DeepSeek-style report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
    print(f"🎨 Features: Modular templates, Clean separation of concerns")
    return html_content

# Usage example for OmicVerse:
# html_report = generate_scRNA_report(adata, species='human', sample_key='sample_id')
# This creates a professional MultiQC-style HTML report with:
# - Clean white theme with dark mode toggle
# - OmicVerse logo integration
# - Responsive design and smooth animations
# - Enhanced navigation and modern UI components
# - Support for all OmicVerse lazy() function features
