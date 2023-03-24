fields_of_interest = {

    'ethnicity': {
        "template":"XXX,",
        "valmap":{ 'Black':"black", 'Asian':'asian', 'White':'white', 'Hispanic':'hispanic' },
        # XXX 'other', nan
        },

    'gender': {
        "template":"XXX,",
        "valmap":{ '0. Male':"male", '1. Female':"female"},
        },

    'age': {
        "template":"XXX years old,",
        "valmap":{ '13. 13':'13', '14. 14':'14', '15. 15':'15', '16. 16':'16', '17. 17':'17', },
        },

#    'religaff': {
#        "template":"Religiously, I consider myself XXX.",
#        "valmap":{ 'Black Prot':"a black protestant", 'none':'unaffiliated', 'evangelical':'an evangelical', 'Jewish':'jewish', 'mainline':'a mainline Protestant', 'Catholic':'catholic', 'Mormon':'mormon' },
#        },

    'religup': {
        "template":"who was raised XXX,",
        "valmap":{},
        },

    'siblings': {
        "template":"with XXX,",
        "valmap":{ '2. 2':'2 siblings', '1. 1':'1 sibling', '0. 0':'no siblings', '3. 3':'3 siblings', '4. 4':'4 siblings', '5. 5':'5 siblings', '6. 6':'6 siblings'},
        },

    'preltrad': {
        "template":"whose parents consider themselves XXX,",
        "valmap":{'3. Black Protestant':'black protestants', '1. Conservative Protestant':'conservative protestants', '4. Catholic':'catholics', '5. Jewish':'jewish', '2. Mainline Protestant':'mainline protestants', '7. Unaffiliated':'unaffiliated', '6. Mormon/LDS':'mormon', '8. Other religion':'members of some other religion', },
        #'9. Indeterminate':
        },

    'parmarital': {
        "template":"whose arents XXX,",
        "valmap":{'1. Married':'are married', '4. Divorced':'are divorced', '2. Living with unmarried partner':'are living with an umarried partner', '5. Separated':'are separated', '6. Never Married':'were never married', '3. Widowed':'widowed' },
        },

    }