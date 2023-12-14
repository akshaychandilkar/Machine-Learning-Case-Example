import pandas as pd
import xlsxwriter

data = [{'Name':'PPA','Duration':3,'Fees':10500},{'Name':'Angular','Duration':3,},{'Name':'Python','Fees':10500}]
df = pd.DataFrame(data)
print(df)

writer = pd.ExcelWriter('MarvellousPandas.xlsx',engine='xlsxwriter')

df.to_excel(writer,sheet_name='Sheet1')

writer.close()
