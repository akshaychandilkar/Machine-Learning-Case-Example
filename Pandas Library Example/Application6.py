import os
import fnmatch
from sys import *
import xlsxwriter

def ExcelCreate(Name):
    workbook = xlsxwriter.Workbook(Name)
    worksheet = workbook.add_worksheet()
    
    worksheet.write('A1','Name')
    worksheet.write('B1','College')
    worksheet.write('C1','Mail ID')
    worksheet.write('D1','Mobile')

    workbook.close()

def main():
    print("----Marvellous Infosystems by Piyush Khairnar----") 

    print("Application name :"+argv[0])

    if (len(argv) !=2):
        print("Error : Invalid number of arguments")

    if (argv[1] == "-h") or (argv[1] == "-H"):
        print("This Script is used to create excel file and write data into it" )
        exit()

    if (argv[1] == "-u") or (argv[1] == "-U"):
        print("Usage : ApplictionName Name_Of_File" )
        exit()

    try:
        ExcelCreate(argv[1])

    except Exception:
        print("Error : Invaild Input")

if __name__ == "__main__":
    main()