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

# Laser Embedding
from laserembeddings import Laser
path_to_bpe_codes = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fcodes'
path_to_bpe_vocab = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/93langs.fvocab'
path_to_encoder = r'/Users/manirathinams/opt/anaconda3/lib/python3.9/site-packages/laserembeddings/data/bilstm.93langs.2018-12-26.pt'
laser = Laser(path_to_bpe_codes, path_to_bpe_vocab, path_to_encoder) 
## loading model from joblib file
classifier=joblib.load('/Users/manirathinams/Documents/KT/pdf practise/Magnum.sav')

##Extracting each page tables using pdfplumber package
def tables_extraction(pdf_path,p):
    p=p-1
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[p]
        tables=first_page.extract_tables()
##Extracting required tables based on their header values & append them all in single list calles dataframes
    dataframes=[]
    for i in range(0, len(tables)):
        for ind,val in enumerate(tables[i]):
            #print(ind,val,i)
            # 1st page
            if ind==0 and 'Language' in val and 'Description' in val:
                dataframes.append(pd.DataFrame(tables[i]))
            # 2nd page
            if ind==0 and 'Product Name' in val:
                dataframes.append(pd.DataFrame(tables[i]))
            elif ind==0 and'Purpose' in val[0]:
                dataframes.append(pd.DataFrame(tables[i]))
    return dataframes

def key_values(dataframes):
    keys1=[]
    val=[]
    ##In list of all dataframes try to get keys and values in a dictionay format
    for df in dataframes:
        for i in range(1, len(df)):
            ## if dataframe is not equalto purpose will follow this code, header w'll be key and row will be values
            if df.iloc[0,0]!='Purpose':
                keys1.append(df.iloc[0,1])
                val.append({classify(str(df.iloc[i,1]))[0]:df.iloc[i,1]})
            ## if dataframe is purpose then 1st col willbe key & 2nd col will be values
            else:
                keys1.append(df.iloc[i,0])
                val.append({classify(str(df.iloc[i,1]))[0]:df.iloc[i,1]})
    ##here we get appended key values in loop using predict_proba to find probability of occurances of each target value
    key=[]
    for x in keys1:
        prob=classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0]
        prob.sort()
    ## we get the max probability occuring value and check whether it's above 0.65% of chance of occuring particular target else we choose the target variable as unmapped        
        if prob[-1]>0.65:
    ##Converting the key to vector format through laser & then predict the target variable for that particular vector value using already trained model 'Classifier'        
            key.append(str((classifier.predict(laser.embed_sentences(x, lang='en')))[0]))
        else:
            key.append('UNMAPPED')
    
    return key, val

def content_classification_slicing(pdf_path, p):
    global a, b
    a=0 
    b=0
    p=p-1
    from pdf2docx import parse
    word_file='/Users/manirathinams/Documents/KT/pdf practise/Magum/Magnum.docx'
    parse(pdf_path, word_file, pages=[p] )
    result = mammoth.convert_to_html(word_file)
    html_file=result.value
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html_file, 'html.parser')
    # print(soup.prettify())
    #Extracting information betwn 'Ingredient & Additional declaration' using enumerate function & get a index value
    temp=[str(tag) for tag in soup.find_all('p')]
    for ind,line in enumerate(temp):
        if 'Ingredient and Allergens' in line:
            a=ind
        elif 'Additional Declarations' in line:
            b=ind

    text=temp[a:b]
    return text

def Content_classification(text):
    ky=[]
    vl=[]
    #taking each line text as a key by removing <strong> & <p> tag, feed them as a X in GS1 element against target Y value
    for t in text:
        x=t.replace('<strong>','').replace('</strong>','').replace('<p>','').replace('</p>','')
        if x not in ['Target Location/Language:','Ingredient List:','Allergen\\Diet List:']:
    #Replacing each lines <strong> tag with '&lt;b&gt;' to capture bold text & keep them as values 
            v=(t.replace('<strong>','&lt;b&gt;').replace('</strong>','&lt;b&gt;').replace('<p>','').replace('</p>',''))
            key_proba=classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0]
            key_proba.sort()
            if key_proba[-1]>0.65:
                ky.append(classifier.predict(laser.embed_sentences(x, lang='en'))[0])
                vl.append({classify(str(v))[0]:v})
            else:
                ky.append('UNMAPPED')
                vl.append(v)
    ##Appending keys and values in key1&val1 where except unmapped keys 
    key1=[ky[k] for k in range(len(ky)) if ky[k]!='UNMAPPED']
    val1=[vl[k] for k in range(len(ky)) if ky[k]!='UNMAPPED']
    return key1, val1

def magnum_main(pdf_path, p):
    dataframes=tables_extraction(pdf_path, p)
    key,val=key_values(dataframes)
    text=content_classification_slicing(pdf_path, p)
    key1,val1=Content_classification(text)
    keys=key+key1
    value=val+val1
    
    general={}
    for i in range(len(keys)):
    ##setdefault will allow duplicate keys in dictionary
        general.setdefault(keys[i], []).append(value[i])
    
    output={str(p):general}
    return output

def Nutrition_slicing(Nutri_file,pgs):
    #pgs=0
    tables = camelot.read_pdf(Nutri_file, pages=str(pgs), flavor='stream')
    data=tables[0].df
    #Extracting Nutritional informations like Protein, Energy upto sodium
    hd=0
    ft=0
    col=0
    for i in range(len(data)):
        for j in range(len(data.columns)):
            if 'Por 100' in str(data.iloc[i,j]):
                hd=i
            elif 'NO provenientes' in str(data.iloc[i,j]):
                ft=i
            elif 'Por porción' in str(data.iloc[i,j]):
                col=j
                
    nutr=data.iloc[hd+1:ft,:col+1]
    ## Extracting 1st four rows seperately to work in Net_contents
    hd1=0
    ft1=0
    col=0
    for i in range(len(data)):
        for j in range(len(data.columns)):
            if 'NUTRIMENTAL' in str(data.iloc[i,j]):
                hd1=i
            elif 'Por 100' in str(data.iloc[i,j]):
                ft1=i
                col=j

    nutr1=data.iloc[hd1+1:ft1,:j]
    nutr_hd=nutr1.dropna(how='all',axis=1)
    return nutr, nutr_hd

## writing a function to print nutrional information in specific format
def nutr_format(nutr):
    value=[]
    for i in range(len(nutr)):
        temp=[]
        for j in range(1, len(nutr.columns)):
            temp.append({'Value':{classify(str(nutr.iloc[i,j]))[0]:nutr.iloc[i,j].replace('\n','')}})
        value.append(temp)
    
    head=[]
    for i in range(len(nutr)):
        x=nutr.iloc[i,0]
        proba=classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0]
        proba.sort()
        if proba[-1] >0.65:
            head.append((classifier.predict(laser.embed_sentences(x, lang='en')))[0])
        else:
            head.append('UNMAPPED')
                    

    final={h:v for h,v in (zip(head, value))}
    return final

##Extract a elements which contains colon seperately, then split the string into two parts using delimiter ':'
##example ['Tamaño de la porción: 48,7 g'] to ['Tamaño de la porción', '48,7 g'] 
def net_content(nutr_hd):    
    content=[nutr_hd.iloc[i,0] for i in range(len(nutr_hd)) if ':' in str(nutr_hd.iloc[i,0])]
    ##seperate a string using delimiter ':' using re.split()
    temp=[]
    for a in content:
        txt=re.split(r':', a.replace('\n',''))
        temp.append(txt)

    ##Writing code to bring dictionary in json format    
    val=[]
    hed=[]
    for k in temp:
        val.append([{'Value':{classify(str(k[1]))[0]:k[1]}}])
        x=k[0]
        proba=classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0]
        proba.sort()
        if proba[-1] >0.65:
            hed.append((classifier.predict(laser.embed_sentences(x, lang='en')))[0])
        else:
            hed.append('UNMAPPED')

    ## Extracting the information 'Contenido energ tico por envase' seperately
    hed1=[]
    val1=[]
    for i in range(len(nutr_hd)):
        for j in range(1, len(nutr_hd.columns)):
            if 'kcal' in str(nutr_hd.iloc[i,j]):
                val1.append([{'Value':{classify(str(nutr_hd.iloc[i,j]))[0]:nutr_hd.iloc[i,j]}}])
                x=nutr_hd.iloc[i,0]
                proba=classifier.predict_proba(laser.embed_sentences(x, lang='en'))[0]
                proba.sort()
                if proba[-1] >0.65:
                    hed.append((classifier.predict(laser.embed_sentences(x, lang='en')))[0])
                else:
                    hed.append('UNMAPPED')

    hd=hed+hed1
    value=val+val1
    nut1={h:v for h,v in zip(hd,value)}
    return nut1

def Nutrition_main(Nutri_file,pgs):
    nutr, nutr_hd=Nutrition_slicing(Nutri_file,pgs)
    Netcontant=net_content(nutr_hd)
    if 'SERVING_SIZE' in Netcontant:
        Netcontant['NUTRITION_FACTS']=[nutr_format(nutr)]
        Nutr_out={pgs:Netcontant}
    else:
        Nutr_out={}
        
    return Nutr_out

def final_extraction(pdf_path, Nutri_file, pgs):
    try:
        out1=magnum_main(pdf_path, pgs)
        out2=Nutrition_main(Nutri_file, pgs)
        final={**out1, **out2}
    except: 
        final=Nutrition_main(Nutri_file, pgs)
        
    return final

output={}
pdf=pdfplumber.open(pdf_path)
for i in range(len(pdf.pages)):
    output.update(final_extraction(pdf_path, Nutri_file, i+1))
{pdf_path:output}
pdf_path='/Users/manirathinams/Documents/KT/pdf practise/Magum/Magnum.pdf'
Nutri_file='/Users/manirathinams/Documents/KT/pdf practise/Magum/Magnum.pdf'
