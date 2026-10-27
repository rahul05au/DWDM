from utils.ui import load_css,show_tree
load_css()

import streamlit as st
from backend.visualization import charts as ch
st.title('8. Visualization')
MAX_ROWS=100_000
WH='SQLite warehouse (database)'

@st.cache_data(show_spinner='Reading table from SQLite…')
def load_warehouse_table(table,signature,limit):
    """Cached read; `signature` (file mtime) invalidates the cache whenever the DB changes."""
    from backend.database.database import read_table
    return read_table(table,limit)

raw,proc=st.session_state.get('raw_df'),st.session_state.get('processed_df')
db_err,tables=None,[]
try:
    from backend.database.database import list_data_tables,table_row_count,db_signature,db_path
    tables=list_data_tables()
except Exception as e:
    db_err=str(e)

opts={}
if tables:opts[WH]=None
if raw is not None:opts['Raw dataset']=raw
if proc is not None:opts['Processed / ETL dataset']=proc
mining=any(st.session_state.get(k) is not None for k in ('classification','kmeans','elbow','rules'))
if db_err:st.error(f'Could not connect to the SQLite warehouse: {db_err}')
if not opts and not mining:
    st.warning('No data available yet. Upload a dataset (page 1) and run ETL (page 3) to load it into the SQLite warehouse, then come back here.');st.stop()

df=None
if opts:
    # Prefer the warehouse (single source of truth) when it has tables
    src=st.radio('Data source',list(opts),horizontal=True)
    if src==WH:
        a,b,c=st.columns([3,1,1])
        last=st.session_state.get('last_etl_table')
        idx=tables.index(last) if last in tables else 0
        table=a.selectbox('Warehouse table',tables,index=idx,key='wh_table')
        if c.button('🔄 Refresh',help='Reload from the database'):load_warehouse_table.clear();st.rerun()
        try:
            total=table_row_count(table)
            df=load_warehouse_table(table,db_signature(),MAX_ROWS)
            b.metric('Rows',f'{total:,}')
            st.caption(f'Connected to {db_path()} · table “{table}” · {len(df.columns)} columns'+(f' · showing first {MAX_ROWS:,} rows' if total>MAX_ROWS else ''))
            if df.empty:st.info(f'Table “{table}” is empty. Re-run ETL to load data.');df=None
        except Exception as e:
            st.error(f'Could not read table “{table}” from SQLite: {e}');df=None
    else:
        df=opts[src]
elif not db_err:
    st.info('The SQLite warehouse has no tables yet – run ETL first. Showing mining results only.')

if df is None:
    st.info('Pick a data source with data to see distribution / category / relationship charts.')
    num,cat,catlike=[],[],[]
else:
    from backend.preprocessing.preprocessing import infer_types
    df=infer_types(df)   # fixes numbers saved as text in older tables
    num=df.select_dtypes('number').columns.tolist()
    cat=[c for c in df.columns if c not in num]
    # low-cardinality numeric columns (e.g. 0/1 flags) also make sense as categories
    catlike=cat+[c for c in num if df[c].nunique()<=12]
    if len(df.columns)==len(num) and not cat:
        st.info('This dataset is fully numeric (for example after one-hot encoding), so text-category charts use low-cardinality numeric columns. Switch the data source above for the original categories.')
if df is not None:
    with st.expander('Column types detected'):
        st.dataframe(df.dtypes.astype(str).rename('type').reset_index().rename(columns={'index':'column'}),width='stretch')
t_dist,t_cat,t_rel,t_mine=st.tabs(['Distributions','Categories','Relationships','Mining results'])
with t_dist:
    if df is not None:
        st.plotly_chart(ch.missing_values(df),width='stretch',key='v_missing')
    if not num:st.info('No numeric columns in this dataset (all columns are text/categories). Use the Categories tab, or check column types below.')
    else:
        a,b=st.columns(2)
        c=a.selectbox('Numeric column',num,key='h');by=b.selectbox('Colour / group by (optional)',[None]+catlike,key='hb')
        a,b=st.columns(2)
        a.plotly_chart(ch.histogram(df,c,by),width='stretch',key='v_hist');b.plotly_chart(ch.boxplot(df,c,by),width='stretch',key='v_box')
        a,b=st.columns(2)
        a.plotly_chart(ch.violin(df,c,by),width='stretch',key='v_violin');b.plotly_chart(ch.density(df,c,by),width='stretch',key='v_density')
with t_cat:
    if not catlike:st.info('No categorical columns available.')
    else:
        c=st.selectbox('Categorical column',catlike,key='c');a,b=st.columns(2)
        a.plotly_chart(ch.bar_counts(df,c),width='stretch',key='v_bar');b.plotly_chart(ch.pie_counts(df,c),width='stretch',key='v_pie')
        if len(catlike)>1:
            g=st.selectbox('Group by (second categorical column)',[x for x in catlike if x!=c],key='cg');a,b=st.columns(2)
            a.plotly_chart(ch.grouped_bar(df,c,g),width='stretch',key='v_gbar');b.plotly_chart(ch.stacked_bar(df,c,g),width='stretch',key='v_sbar')
        if num:
            a,b=st.columns(2);m=a.selectbox('Numeric measure',num,key='cm');ag=b.selectbox('Aggregation',['mean','sum','median','max','min'],key='ca')
            st.plotly_chart(ch.summary_bar(df,c,m,ag),width='stretch',key='v_sum')
with t_rel:
    if len(num)>1:
        st.plotly_chart(ch.correlation_heatmap(df),width='stretch',key='v_corr')
        a,b,c=st.columns(3);x=a.selectbox('X',num,key='sx');y=b.selectbox('Y',num,index=1,key='sy');col=c.selectbox('Colour',[None]+catlike,key='sc')
        st.plotly_chart(ch.scatter(df,x,y,col),width='stretch',key='v_scatter')
        st.plotly_chart(ch.line_chart(df,x,y,col),width='stretch',key='v_line')
        sm=st.multiselect('Scatter matrix columns',num,default=num[:4],key='smc')
        if len(sm)>1:st.plotly_chart(ch.scatter_matrix(df,sm,col),width='stretch',key='v_matrix')
        if len(num)>2:
            a,b,c=st.columns(3);x3=a.selectbox('3D X',num,index=0,key='x3');y3=b.selectbox('3D Y',num,index=1,key='y3');z3=c.selectbox('3D Z',num,index=2,key='z3')
            st.plotly_chart(ch.scatter3d(df,x3,y3,z3,col),width='stretch',key='v_3d')
            pc=st.multiselect('Parallel-coordinates columns',num,default=num[:5],key='pcc')
            if len(pc)>1:
                cc=col or pc[0]
                st.plotly_chart(ch.parallel_coordinates(df,[p for p in pc if p!=cc] or pc,cc),width='stretch',key='v_par')
    else:st.info('Need at least two numeric columns.')
with t_mine:
    cls = st.session_state.get('classification')
    km = st.session_state.get('kmeans_result') or st.session_state.get('kmeans')
    el = st.session_state.get('elbow_result', {}).get('df') if isinstance(st.session_state.get('elbow_result'), dict) else st.session_state.get('elbow')
    ru = st.session_state.get('rules')
    if not any([cls, km is not None, el is not None, ru is not None and not ru.empty]):
        st.info('Run something on the Data Mining page first (classification, clustering or association rules).')
    if cls:
        st.subheader(f"Classification – {cls['algorithm']} → {cls['target']}")
        a, b = st.columns(2)
        a.plotly_chart(ch.metrics_bar(cls['metrics']), use_container_width=True)
        b.plotly_chart(ch.confusion_heatmap(cls['metrics']['Confusion Matrix'], cls['classes']), use_container_width=True)
        if cls.get('importances') is not None:
            st.plotly_chart(ch.importance_bar(cls['importances']), use_container_width=True)
        if cls.get('tree_model') is not None:
            st.subheader('Decision tree')
            show_tree(cls, 'viz')
    if km is not None:
        st.subheader('Clustering (K-Means)')
        p_mat = km['p'] if isinstance(km, dict) else km[1]
        lab_vec = km['labels'] if isinstance(km, dict) else km[2]
        st.plotly_chart(ch.cluster_scatter(p_mat, lab_vec), use_container_width=True)
    if el is not None:
        st.plotly_chart(ch.elbow_chart(el), use_container_width=True)
    if ru is not None and len(ru):
        st.subheader('Association rules')
        st.plotly_chart(ch.rules_scatter(ru), use_container_width=True)
