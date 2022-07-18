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

#Laser Embeddings
from laserembeddings import Laser
path_to_bpe_codes = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fcodes'
path_to_bpe_vocab = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fvocab'
path_to_encoder = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/bilstm.93langs.2018-12-26.pt'
laser = Laser(path_to_bpe_codes, path_to_bpe_vocab, path_to_encoder) 

mlp_model=joblib.load('/Users/manirathinams/Documents/Python /woolsworth pdf/woolworths.sav')
classifier=joblib.load('/Users/manirathinams/Documents/Python /woolsworth pdf/woolworth_Nutrition.pkl')

def data_extraction_table(file_path, pgs):
    from pdf2docx import parse
    word_file='/Users/manirathinams/Documents/KT/woolsworth_Pdf/AIFWW.docx'
    parse(file_path,word_file, pages=[pgs-1])
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
    return single_line_list, col_row_span

#Remove empty strings in a list
def remove_empty_str_list(lists):
    temp=[s for s in lists if s]
    return temp

def general_dict_content(single_line_list, col_row_span):
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
    return general_dict

def nutrition_content_extraction(file_path, pgs):
    with pdfplumber.open(file_path) as pdf:
        single_pg=pdf.pages[pgs-1]
        text=single_pg.extract_text().split('\n')
#Extracting nutrition & servings content from multiple pages in pdf
    match_text = ('energy', 'calories', 'total carbohydrate', 'dietary fiber', 'cholesterol', 'fat', 'total fat', 'sodium',
            'protein', 'saturated', 'trans', 'polyunsaturated', 'omega', 'acids', 'epa', 'dha', 'monounsaturated',
            'carbohydrate', 'sugars', 'vitamin d', 'calcium', 'iron', 'potassium', 'total sugars', 'sodium',
            'saturated fat', 'trans fat')

    extract_nutri = []
    start_Ingredient = False
    if 'Nutrition Information' in ''.join(text):
        start = False
        for i in range(len(text)):
            if 'Back of Pack  Nutrition' in text[i]:
                start = True
            if '*Percentage daily intakes' in text[i]:
                start = False
            if start:
                if text[i].strip() =='' or text[i]==None:
                    continue
                extract_nutri.append(text[i])
    elif 'INGREDIENTS LIST' in ''.join(text):
        for j in range(len(text)):
            if 'INGREDIENTS LIST' in text[j]:
                sp = re.findall(r'[A-Za-z]+',text[j+1].replace("-", "").strip())
                if sp[0].lower().strip() in match_text:
                    start_Ingredient = True
            if '*Percentage daily intakes' in text[j]:
                start_Ingredient = False
            if start_Ingredient:
                if text[j].strip() =='' or text[j]==None:
                    continue
                extract_nutri.append(text[j])
    #find the row with val 'cal' then add calories in front of that row 
    nutri_list=[]
    for k in range(len(extract_nutri)):
        if re.search(r'\bcal\b',extract_nutri[k].lower()):
            nutri_list.append('Calories ' + extract_nutri[k])
        else:
            nutri_list.append(extract_nutri[k])
    return nutri_list

def nutrition_serving(nutri_list):
    Header_dic={}
    nutrition=[]
    serving=[]
    for i in  range(len(nutri_list)):
        first_ele=re.findall(r'[\-\sA-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]+',nutri_list[i])
        prob=(mlp_model.predict_proba(laser.embed_sentences(first_ele[0], lang='en'))[0])
        prob.sort()
        predict=mlp_model.predict(laser.embed_sentences(first_ele[0], lang='en'))
        classified_output=predict
        if prob[-1]>=0.85 and classified_output in ('SERVING_SIZE', 'SERVING_PER_CONTAINER'):
            serving.append(nutri_list[i])
        else:
            first_elem=first_ele[0].replace("-", "").replace("EPA", "Omega").replace("DHA", "Omega").replace("DHA","Omega").replace("Includes", "Includes Added Sugars")
            proba=(classifier.predict_proba(laser.embed_sentences(first_elem,lang='en'))[0])
            proba.sort()
            predict=classifier.predict(laser.embed_sentences(first_elem, lang='en'))
            classified_output=predict 
            if proba[-1]>=0.90:
                nutrition.append(nutri_list[i])
    return nutrition, serving

def nutrition_correct_list(nutrition):
    clean_nutrition=[]
    #Here nutrition will be like this ['Energy 472 kJ 5% 497 kJ'] using regex will seperate each elements
    for x in nutrition:
        element=[]
        regex_extracted = re.findall(
                r"([\w\,\-\s]*?)\s+(\<?\s?\-?\d{0,3}\.?\d{0,2}\s?(%|g added sugars|g|kj|kcal|mg|mcg|cal))", x, flags=re.I)
        #using regex findall, will get like this [('Sodium', '258 mg', 'mg'), ('', '11%', '%'), ('', '272 mg', 'mg')]
        if not regex_extracted:
            regex_extracted = re.findall(r"([\w\,\-\s]*?)\s+((\<?\s?\-?\d{0,3}\.?\d{0,2}\s?))", x, flags=re.I)
        col=[]
        #here we getting tuple elements like energy in elementlist and values in seperate colList, finally merge col & element list in clean_nutrition
        for tuple_cnt in regex_extracted:
            if tuple_cnt[0] and tuple_cnt[0].strip() not in ("-"): 
                element.append(tuple_cnt[0])
            if tuple_cnt[1]:
                col.append(tuple_cnt[1])
        if col:
            element.extend(col)
        clean_nutrition.append(element)
    return clean_nutrition

def nutrition_classification(clean_nutrition):
    key=[]
    val=[]
    #getting keys,val from clean_nutrition based on classifier model
    for row in clean_nutrition:
        text=str(row[0].replace('-','').strip())
        prob=(classifier.predict_proba(laser.embed_sentences(text, lang='en'))[0])
        prob.sort()
        if prob[-1]>=0.90:
            predicted=classifier.predict(laser.embed_sentences(text, lang='en'))[0]
            key.append(predicted)
        cells=[]
        for t in range(1,len(row)):
            if '%' in str(row[t]):
                cells.append({'PDV':{classify(row[t])[0]:str(row[t]).replace('<','&lt;').replace('>', '&gt;')}})
            else:
                cells.append({'Value':{classify(row[t])[0]:str(row[t]).replace('<','&lt;').replace('>', '&gt;')}})
        if cells:
            val.append(cells)

    nutrition={}
    for m in range(len(key)):
        nutrition.setdefault(key[m],[]).extend(val[m])
    return nutrition

def serving_content(serving):
    serving_lst=[]
    for x in range(0,len(serving)):
        if ':' in str(serving[x]):
            serve=serving[x].split(':')
        else:
            serve=[serving[x]]
        serving_lst.append(serve)   

    serv_dic={}
    for s in range(len(serving_lst)):
        if len(serving_lst[s])>1:
            for n in range(1, len(serving_lst[s])):
                serving_lst[s][n]=serving_lst[s][n].replace('<','&lt;').replace('>','&gt;')
                if serving_lst[s][n]!='':
                    if serving_lst[s][0].strip() in serv_dic:
                        serv_dic[serving_lst[s][0].strip()].append({classify(serving_lst[s][n])[0]:str(serving_lst[s][n]).strip()})
                    else:
                        serv_dic[serving_lst[s][0].strip()]=[{classify(serving_lst[s][n])[0]:str(serving_lst[s][n]).strip()}]
        elif len(serving_lst[s])==1:
            if serving_lst[s][0].strip() in serv_dic:
                serv_dic[serving_lst[s][0].strip()].append({classify(serving_lst[s][0])[0]:str(serving_lst[s][0]).strip()})
            else:
                serv_dic[serving_lst[s][0].strip()]=[{classify(serving_lst[s][0])[0]:str(serving_lst[s][0]).strip()}]

    serving_dic={}
    for ser_key, ser_val in serv_dic.items():
        #print(ser_key,'\n')
        proba=(mlp_model.predict_proba(laser.embed_sentences(ser_key,lang='en'))[0])
        proba.sort()
        predict=mlp_model.predict(laser.embed_sentences(ser_key,lang='en'))[0]
        classified_output=predict
        if proba[-1]>=0.85:
            if classified_output!='None':
                if classified_output in serving_dic:
                    serving_dic[classified_output].append(ser_val)
                else:
                    serving_dic[classified_output]=ser_val
            else:
                if ser_key in serving_dic:
                    serving_dic[ser_key].append(ser_val)
                else:
                    serving_dic[ser_key]=ser_val
    return serving_dic

def general_main(file_path, pgs):
    single_line,col_row_sp=data_extraction_table(file_path, pgs)
    general=general_dict_content(single_line, col_row_sp)
    return general

def nutrition_main(file_path, pgs):
    nutri_list=nutrition_content_extraction(file_path, pgs)
    nutrition, serving=nutrition_serving(nutri_list)
    clean_nutrition=nutrition_correct_list(nutrition)
    serving_dic=serving_content(serving)
    if serving_dic:
        serv=serving_dic
    else:
        serv={}
    nutrition_dic=nutrition_classification(clean_nutrition)
    if nutrition_dic:
        nutr={"NUTRITION_FACTS":nutrition_dic}
    else:
        nutr={}
    final={**serv,**nutr}
    return final

def woolworths_main(file_path, pgs):
    out1=general_main(file_path, pgs)
    out2=nutrition_main(file_path, pgs)
    output={**out1,**out2}
    final={str(pgs):output}
    return final

file_path='/Users/manirathinams/Documents/KT/woolsworth_Pdf/AIFWW Potato Mashie with Gravy 200g2022 - 200g AUSTRALIAN INTERNATIONAL FOODS PTY LTD.pdf'
result=woolworths_main(file_path,4)
result
