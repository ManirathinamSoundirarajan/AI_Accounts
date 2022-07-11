import pdf2docx
import pandas as pd
import numpy as np
from langid import classify
import camelot
import string
import re
import tabula
import mammoth
import pdfplumber
import joblib
from bs4 import BeautifulSoup
________________________________________________________________
from laserembeddings import Laser
path_to_bpe_codes = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fcodes'
path_to_bpe_vocab = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fvocab'
path_to_encoder = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/bilstm.93langs.2018-12-26.pt'
laser = Laser(path_to_bpe_codes, path_to_bpe_vocab, path_to_encoder) 

classifier=joblib.load('/Users/manirathinams/Documents/Python /woolsworth pdf/sansbury_nutri.sav')
mlp_model=joblib.load('/Users/manirathinams/Documents/Python /woolsworth pdf/woolworths.sav')
________________________________________________________________

def data_extraction_table(file_path, pgs):
    from pdf2docx import parse
    word_file='/Users/manirathinams/Documents/KT/sansbury_pdf/Input_files/JS.docx'
    parse(file_path, word_file, pages=[pgs-1])
    result=mammoth.convert_to_html(word_file).value
    soup=BeautifulSoup(result, 'html.parser')
    #Extracting data of each cells & merged cells in a row to a table format using row_span, column_span
    col_row_span=[]
    single_line_list=[]
    for table in soup.find_all('table'):
        rows=[]
        for row in table.find_all('tr'):
            cells=[]
            for cell in row.find_all('td'):
                if cell.has_attr('colspan'):
                    raw_html=str(cell).replace('<strong>','start_bold').replace('</strong>','end_bold').replace('</p>','\n').replace("<br>",'\n').replace('br/','\n').strip()
                    #print('\n',BeautifulSoup(raw_html, 'html.parser').text)
                    rows.append((BeautifulSoup(raw_html, 'lxml').text).replace('<', '&lt;').replace('>', '&gt;').replace('start_bold', '<b>').replace('end_bold', '</b>'))
                elif cell.has_attr('rowspan'):
                    raw_html=str(cell).replace('<strong>','start_bold').replace('</strong>','end_bold').replace('</p>','\n').replace("<br>",'\n').replace('br/','\n').strip()
                    #print('\n',BeautifulSoup(raw_html, 'html.parser').text)
                    rows.append((BeautifulSoup(raw_html, 'lxml').text).replace('<', '&lt;').replace('>', '&gt;').replace('start_bold', '<b>').replace('end_bold', '</b>'))
                else:
                    raw_html=str(cell).replace('<strong>','start_bold').replace('</strong>','end_bold').replace('</p>','\n').replace("<br>",'\n').replace('br/','\n').strip()
                    cells.append((BeautifulSoup(raw_html, 'lxml').text).replace('<', '&lt;').replace('>', '&gt;').replace('start_bold', '<b>').replace('end_bold', '</b>'))
            normal_cntnt=remove_empty_str_list(cells)
            if len(normal_cntnt)>0:
                single_line_list.append(normal_cntnt)
        col_row=remove_empty_str_list(rows)
        if len(col_row)>0:
            col_row_span.extend(col_row)
    return single_line_list,col_row_span 

#Remove empty strings in a list
def remove_empty_str_list(lists):
    temp=[s for s in lists if s]
    return temp
________________________________________________________________

def general_dict_content(single_line_list,col_row_span,pgs):
    general_dict={}
    unwanted_text = ('supportingtext', 'pack copy', 'secondary facing panel', 'use on pack', 'primary facing panel', 
    'component', 'icon', '-', 'text', 'as per design brief', 'supporting text', 'recycling icons')
    key1,val1=[],[]
    for each_row in single_line_list:
        if len(each_row)>=2:
        #Here if 1st element in ky_brand list, then it will be brand, subbrand, prod legal name keys and respective values
            ky_brand=str(each_row[0]).lower().strip().replace('<b>','').replace('</b>','').replace('\n','').replace(':','')
            if ky_brand in ["brand", "brand type", "product legal name", "any other information (bop)"]:
                prob=(mlp_model.predict_proba(laser.embed_sentences(ky_brand, lang='en'))[0])
                prob.sort()
                predict=mlp_model.predict(laser.embed_sentences(ky_brand, lang='en'))[0]
                classified_output=predict
                if prob[-1]>=0.85:
                    key1.append(classified_output)
                    val1.append({classify(each_row[1])[0]:str(each_row[1]).replace('\n','')})
            else:
            #Here 1st IF condn is used for content classification of first element, 2nd ELIF is used for classification of key element, 3rd and nested ELSE used for UNMAPPED part
                first_element=str(each_row[0].lower().strip()).replace('<b>','').replace('</b>','').replace('\n','')
                prob=(mlp_model.predict_proba(laser.embed_sentences(first_element, lang='en'))[0])
                prob.sort()
                predict=mlp_model.predict(laser.embed_sentences(first_element, lang='en'))[0]
                classified_output=predict
                each_row[1]=str(each_row[1].lower().strip()).replace('<','&lt;').replace('>','&gt:').replace('\n','')
                if classified_output in ("VARIANT", "WARNING_STATEMENTS")and prob[-1]>=0.95:
                    val1.append({classify(first_element)[0]:str(first_element)})
                    key1.append(classified_output)
                elif prob[-1]>=0.85 and classified_output not in ["NUTRI_TABLE_HEADERS","SERVING_PER_CONTAINER"]:
                    if each_row[1]!='':
                        val1.append({classify(each_row[1])[0]:str(each_row[1])})
                        key1.append(classified_output)
                else:
                    if str(each_row[1]).lower().strip().replace('<b>','').replace('</b>','').replace('\n','') not in unwanted_text:
                        row_text=each_row[1].replace('<','&lt;').replace('>','&gt:').replace('\n','').strip()
                        classified_output='UNMAPPED'
                        if classified_output in  general_dict:
                            general_dict[classified_output].append({classify(row_text)[0]:str(row_text)})
                        else:
                            general_dict[classified_output]=[{classify(row_text)[0]:str(row_text)}]
        else:
            if str(each_row).lower().strip().replace('<b>','').replace('</b>','').replace(':','').replace('\n','') not in unwanted_text:
                each_row[0]=str(each_row[0]).replace('<','&lt;').replace('>','&gt:').replace('\n','').strip()
                classified_output="UNMAPPED"
                if classified_output in general_dict:
                    general_dict[classified_output].append({classify(each_row[0])[0]:str(each_row[0])})
                else:
                    general_dict[classified_output]=[{classify(each_row[0])[0]:str(each_row[0])}]
    key2,val2=[],[]
    for x in range(0, len(col_row_span)):
#Here it will get values from col_row_span, 1st IF condn to get serial_no, 2nd ELIF condn to ALLERGEN, INGREDIENTS, 3rd ELSE for UNMAPPED keys& values
        row_text=str(col_row_span[x]).replace('<b>','').replace('</b>','').replace('\n','')
        prob=mlp_model.predict_proba(laser.embed_sentences(row_text, lang='en'))[0]
        prob.sort()
        predict=mlp_model.predict(laser.embed_sentences(row_text ,lang='en'))[0]
        classified_output=predict
        if prob[-1]>=0.95 and classified_output=="SERIAL_NUMBER":
            key2.append(classified_output) 
            val2.append({classify(col_row_span[x])[0]:str(col_row_span[x]).replace('\n','')})
        elif prob[-1]>=0.85 and classified_output not in ("SERIAL_NUMBER","NUTRI_TABLE_HEADERS"):
            key2.append(classified_output)
            val2.append({classify(col_row_span[x])[0]:str(col_row_span[x]).replace('<','&lt:').replace('>','&gt:').replace('\n','').strip()})
        else:
            if str(col_row_span[x].lower().strip()).replace('<b>','').replace('</b>','').replace('\n','') not in unwanted_text:
                classified_output='UNMAPPED'
                if classified_output in general_dict:
                    general_dict[classified_output].append({classify(col_row_span[x])[0]:str(col_row_span[x])})
                else:
                    general_dict[classified_output]=[{classify(col_row_span[x])[0]:str(col_row_span[x])}]                
    keys=key1+key2
    val=val1+val2
    for i in range(len(keys)):
        general_dict.setdefault(keys[i],[]).append(val[i])
    #output={str(pgs):general_dict}    
    return general_dict
________________________________________________________________

def bop_nutrition_extraction(file_path,pgs):
#Extracting the BackofPack Nutrition table from multiple tables in a page using Camelot
    tables = camelot.read_pdf(file_path, pages=str(pgs), line_scale=40 ,flavor='lattice')    
    bop=0
    bop_hd=0
    for x in range(len(tables)):
        data=tables[x].df
        for s in range(len(data)):
            for t in range(len(data.columns)):
                if 'back of pack declaration' in str(data.iloc[s,t]).lower().replace('\n','') and 'nutrition' in str(data.iloc[s,t+1]).lower():
                    bop_hd=s
                    bop=x
    dataframes=tables[bop].df
    dfs=dataframes.iloc[bop_hd:,:]
    return dfs
________________________________________________________________

def fop_nutrition_extraction(file_path,pgs):
#Extracting the FrontofPack Nutrition table from multiple tables in a page using Camelot
    tables = camelot.read_pdf(file_path, pages=str(pgs), line_scale=40 ,flavor='lattice')
    fop=0
    fop_hd=0
    fop_ft=0
    for x in range(len(tables)):
        data1=tables[x].df
        for s in range(len(data1)):
            for t in range(len(data1.columns)):
                if 'nutrition information' in str(data1.iloc[s,t]).lower() and 'front of pack declaration' in str(data1.iloc[s+1,t]).lower().replace('\n',''):
                    fop_hd=s+1
                    fop=x
                elif 'of your reference' in str(data1.iloc[s,t]).lower().replace('\n',''):
                    fop_ft=s
    dataframes=tables[fop].df
    df=pd.DataFrame(dataframes.iloc[fop_hd:fop_ft,:])
    dfs=df.T
    return dfs
________________________________________________________________

def nutrition_classification(dfs):
    match_text=['energy','fat','saturates','monounsaturates','polyunsaturates','carbohydrate','sugars','starch','fibre','protein','salt']
#Removing empty cells in a each row of dataframe    
    tab=[]
    for i in range(len(dfs)):
        r=[]
        for j in range(len(dfs.columns)):
            if dfs.iloc[i,j]!='':
                r.append(dfs.iloc[i,j])
        if r:
            tab.append(r)
#Extracting the keys based on match_text & respective values using nested for loop 
    key=[]
    val=[]
    for m in range(len(tab)):
        cell=[]
        for n in range(0,len(tab[m])):
            if str(tab[m][n]).lower().replace('-','').strip() in match_text:
                x=str(tab[m][n]).lower().replace('-','').strip()
                prob=(classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0])
                prob.sort()
                if prob[-1]>0.75:
                    key.append(classifier.predict(laser.embed_sentences(x, lang='en'))[0])
                for p in range(n+1,len(tab[m])):
                    if '%' in str(tab[m][p]): 
                        cell.append({'PDV':{classify(str(tab[m][p]))[0]:(tab[m][p]).replace('<','&lt;').strip()}})
                    else:
                        clean=re.sub('<[A-Za-z].+>',',',tab[m][p])
                        cell.append({'Value':{classify(str(clean))[0]:(clean).replace('<','&lt;').strip()}})
        if cell:
            val.append(cell)
#Removing the duplicate keys
    nutr={}
    for k in range(len(key)):
        nutr.setdefault(key[k],[]).extend(val[k])   
    return nutr
________________________________________________________________

def nutrition_serve_header(file_path, pgs):
#Identify and append the index of both FrontofPack, BackofPack tables into list using camelot from multiple tables in a page 
    tables = camelot.read_pdf(file_path, pages=str(pgs), line_scale=40 ,flavor='lattice')
    multi_tab=[]
    for x in range(len(tables)):
        data=tables[x].df
        for s in range(len(data)):
            for t in range(len(data.columns)):
                if 'back of pack declaration' in str(data.iloc[s,t]).lower().replace('\n','') and 'nutrition' in str(data.iloc[s,t+1]).lower():
                    multi_tab.append(tables[x].df)
                elif 'nutrition information' in str(data.iloc[s,t]).lower() and 'front of pack declaration' in str(data.iloc[s+1,t]).lower().replace('\n',''):
                    multi_tab.append(tables[x].df)
#Removing empty cells in a each row of dataframe
    if multi_tab:
        tab=[]
        for table in multi_tab:
            for i in range(len(table)):
                r=[]
                for j in range(len(table.columns)):
                    if table.iloc[i,j]!='':
                        r.append(table.iloc[i,j])
                if r:
                    tab.append(r)
#Extracting only the serving contents from dataframe by sorting len of row <2 and len of text in each row <20
        ser_key=[]
        ser_val=[]
        for k in range(0,len(tab)):
            if len(tab[k])<=2:
                for g in range(0,len(tab[k])):
                    text=str(tab[k][g]).lower().replace('\n','').strip()
                    if len(text.split())<=20: 
                        prob=(classifier.predict_proba(laser.embed_sentences(text, lang='en'))[0])
                        prob.sort()
                        predict=(classifier.predict(laser.embed_sentences(text, lang='en'))[0])
                        classified_output=predict           
                        if classified_output in ('SERVING_PER_CONTAINER','SERVING_SIZE','NUTRITION_TABLE_CONTENT') and prob[-1]>=0.85:
                            ser_key.append(classified_output)
                            ser_val.append({classify(text)[0]:str(text)})
    #Removing the duplicate keys
        serv_nutr={}
        for k in range(len(ser_key)):
            serv_nutr.setdefault(ser_key[k],[]).append(ser_val[k])
        #output={str(pgs):serv_nutr}
        return serv_nutr
    else:
        return {}
________________________________________________________________

def general_main(file_path, pgs):
#writing a function for general information part    
    single_line, col_row_sp=data_extraction_table(file_path, pgs)
    general=general_dict_content(single_line, col_row_sp, pgs)
    serve=nutrition_serve_header(file_path, pgs)
    if serve:
        sansbury_final={**general,**serve}
        return sansbury_final
    else:
        return general
________________________________________________________________
def nutrition_main(file_path, pgs):
#writing a function for nutritional information part    
    fop_dfs=fop_nutrition_extraction(file_path, pgs)
    fop_nutr=[nutrition_classification(fop_dfs)]
    bop_dfs=bop_nutrition_extraction(file_path, pgs)
    bop_nutr=[nutrition_classification(bop_dfs)]
    fop_nutr.extend(bop_nutr)
    if fop_nutr !=[{},{}]:
        fnl={"NUTRITION_FACTS":fop_nutr}
    else:
        fnl={}             
    return fnl
________________________________________________________________
def final_extraction(file_path,pgs):
    out1=general_main(file_path, pgs)
    out2=nutrition_main(file_path, pgs)
    output={**out1,**out2}
    final={str(pgs):output}
    return final

file_path='/Users/manirathinams/Documents/KT/sansbury_pdf/Input_files/JS Cranberry Cosmopolitan 50cl - 5000000ML (Gabriel Boudier)_V1 (1).pdf'
result=final_extraction(file_path, 3)
result