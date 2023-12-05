class few_shot_settings:
    
    @staticmethod
    def get_prefix():
        return f"""
		You are an agent designed to interact with a Snowflake with schema detail in Snowflake querying about Company Inventory & Item statements. You have to write syntactically correct Snowflake sql query based on a users question.
		No matter what the user asks remember your job is to produce relevant SQL and only include the SQL, not the through process. So if a user asks to display something, you still should just produce SQL.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
		If you don't know the answer, provide what you think the sql should be but do not make up code if a column isn't available. Use snowflake aggregate functions like SUM, MIN, MAX, etc. if user ask to find total, minimum or maximum.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. 
		Few rules to follow are
		1. always interpret QUARTER_NAME in the format YYYY-QQ from various inputs from user for example inputs like Q1'22 or 1st qtr 22 or 2022 quarter1 or 22Q1 or 22'Q1 or 22 Q1 or Q1 of financial year 22 should be translated as YYYY-QQ 
		2. You must interpret next quarter, previous quarter, next year first quarter etc keeping below rule into consideration: 
           a. Rule to determine quarter would be as per below
                for MONTH(CURRENT_DATE()) IN (2, 3, 4) quarter would be Q1 and year would be YEAR(CURRENT_DATE())+1
                for MONTH(CURRENT_DATE()) IN (5, 6, 7) quarter would be Q2 and year would be YEAR(CURRENT_DATE())+1
                for MONTH(CURRENT_DATE()) IN (8, 9, 10) quarter would be Q3 and year would be YEAR(CURRENT_DATE())+1
                for MONTH(CURRENT_DATE()) IN (11, 12) quarter would be Q4 and year would be YEAR(CURRENT_DATE())+1
                for MONTH(CURRENT_DATE()) IN (1) quarter would be Q4 and year would be YEAR(CURRENT_DATE())
           Always calculate value of QUARTER_NAME by using above 2 rules and concatenate to bring in format  'YYYY-QQ'. 
           Strictly do not use inner query for such questions from user. Refer example containing next quarter the QUARTER_NAME is calculated as per Current_date year is 2023 and month is Nov or 11
        3. Always use column aliases as per example and metadata
        4. for any aggregation function like sum, avg, max, min and count, the must be GROUP BY clause on all columns selected without aggregate function. 
        5. preference is to use direct inner or left join. Avoid inner queries in WHERE clause.
        """
    
    @staticmethod
    def get_suffix():
        return """Question: {question}
        Context: {context}

        SQL_cmd: ```sql ``` \n

        """, ["question", "context"]
    
    @staticmethod
    def get_examples():
        examples = [
            {
                "input": "Display Aggregated Amount for Each Quarter.?",
                "sql_cmd": '''SELECT QUARTER_NAME AS "QUARTER NAME", to_varchar(SUM(AMOUNT), '$ 9,999,999,999.90') AS "TOTAL AMOUNT" FROM FINANCIALS.MARVELL_DEMO.INVENTORY_ACTUALS GROUP BY QUARTER_NAME;''',
            },
            {
                "input": "Display me count of distinct products as per BU for SPG Division",
                "sql_cmd": '''SELECT COUNT(*) AS "TOTAL COUNT", BU as "BU", DIVISION AS "DIVISION" FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS WHERE DIVISION  = \'SPG\' GROUP BY BU ,DIVISION;''',
            },
            {
                "input": "Display Total Inventory Amount for each quarter in business unit BBA",
                "sql_cmd": '''SELECT A.QUARTER_NAME AS "QUARTER NAME", A.ITEM_WID AS "ITEM WID",B.BU AS "BU", to_varchar(SUM(A.AMOUNT), '$ 9,999,999,999.90') AS "TOTAL AMOUNT" FROM FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY  A INNER JOIN FINANCIALS.MARVELL_DEMO.ITEM_DETAILS  B ON A.ITEM_WID = B.ITEM_WID WHERE B.BU = 'BBA' GROUP BY B.BU, A.QUARTER_NAME, A.ITEM_WID ORDER BY "QUARTER NAME", "TOTAL AMOUNT";''',
            },
            {
                "input": "Display projected Inventory for BBA organization in YEAR 2021 and Quarter Q4",
                "sql_cmd": '''SELECT A.ITEM_WID AS "ITEM WID", to_varchar(A.AMOUNT, '$ 9,999,999,999.90') AS "AMOUNT" FROM FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY A INNER JOIN FINANCIALS.MARVELL_DEMO.ITEM_DETAILS B ON A.ITEM_WID = B.ITEM_WID WHERE B.BU = \'BBA\' AND A.QUARTER_NAME = \'2021-Q4\';''',
            },
            {
                "input": "For a Particular BU , For Eg : CDSP What's the actual vs In hand vs Projected total amount for 5nm 'Tech Group' or 'technology group' in Year 2024 and Quarter Q3",
                "sql_cmd": '''SELECT B.QUARTER_NAME AS "QUARTER NAME", A.BU AS "BU", A.TECHNOLOGY_GROUP AS "TECHNOLOGY GROUP", to_varchar(SUM(B.AMOUNT), '$ 9,999,999,999.90') AS "TOTAL ACTUAL AMT",
                              to_varchar(SUM(C.AMOUNT), '$ 9,999,999,999.90') AS "TOTAL ON-HAND AMT", to_varchar(SUM(PROJINV.AMOUNT), '$ 9,999,999,999.90') AS "TOTAL PROJECTED AMT" FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS A
                              LEFT JOIN FINANCIALS.MARVELL_DEMO.INVENTORY_ACTUALS  B ON A.ITEM_WID  = B.ITEM_WID 
                              LEFT JOIN FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS C ON A.ITEM_WID  = C.ITEM_WID 
                              LEFT JOIN FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY PROJINV ON A.ITEM_WID  = PROJINV.ITEM_WID 
                              WHERE A.TECHNOLOGY_GROUP  = '5nm' AND A.BU = 'CDSP' AND  B.QUARTER_NAME  = '2024-Q3' GROUP BY ALL;''',
            },
            {
                "input": "Which Division is Projecting the highest Amount or inventory",
                "sql_cmd": '''SELECT B.QUARTER_NAME AS "QUARTER NAME", A.DIVISION AS "DIVISION", to_varchar(SUM(B.AMOUNT), '$ 9,999,999,999.90') "PROJECTED AMOUNT"
                              FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS A LEFT JOIN FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY B
                              ON A.ITEM_WID  = B.ITEM_WID WHERE B.QUARTER_NAME LIKE ANY ('%') GROUP BY ALL
                              QUALIFY  ROW_NUMBER () OVER(ORDER BY "PROJECTED AMOUNT" DESC) =1;''',
            },
            {
                "input": "what's the manufacturing comparison of type of items in q3 2024",
                "sql_cmd": '''SELECT B.QUARTER_NAME AS "QUARTER NAME", A.ITEM_TYPE AS "ITEM TYPE", to_varchar(SUM(B.AMOUNT), '$ 9,999,999,999.90') "TOTAL AMOUNT"
                              FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS A LEFT JOIN FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS B 
                              ON A.ITEM_WID  = B.ITEM_WID WHERE B.QUARTER_NAME = '2024-Q3' GROUP BY "QUARTER NAME", "ITEM TYPE" 
                              ORDER BY "TOTAL AMOUNT" DESC;''',
            },
            {
                "input": "what's the manufacturing comparison on item's quantity of wafer group in q3 2024",
                "sql_cmd": '''SELECT B.QUARTER_NAME AS "QUARTER NAME", B.ITEM_WID AS "ITEM WID", A.PRODUCT_GROUP AS "PRODUCT GROUP", SUM(B.QUANTITY) AS "TOTAL QUANTITY"
                              FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS A LEFT JOIN FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS B 
                              ON A.ITEM_WID  = B.ITEM_WID WHERE B.QUARTER_NAME = '2024-Q3' AND A.PRODUCT_GROUP='WFR' GROUP BY "PRODUCT GROUP", "QUARTER NAME", B.ITEM_WID, B.QUANTITY
                              ORDER BY "TOTAL QUANTITY" DESC;''',
            },
            {
                "input": "what are top ten items in finished goods stage in quarter 3 year 24?",
                "sql_cmd": '''SELECT B.QUARTER_NAME AS "QUARTER NAME", B.ITEM_WID AS "ITEM WID", A.ITEM_STAGE AS "ITEM STAGE", SUM(B.QUANTITY) AS "TOTAL QUANTITY"
                              FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS A LEFT JOIN FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS B 
                              ON A.ITEM_WID  = B.ITEM_WID WHERE B.QUARTER_NAME = '2024-Q3' AND A.ITEM_STAGE='FG' GROUP BY "ITEM STAGE", "QUARTER NAME", B.ITEM_WID, B.QUANTITY
                              ORDER BY "TOTAL QUANTITY" DESC LIMIT 10;''',
            },
            {
                "input": "Show me top 10 values of Item stage FG",
                "sql_cmd": '''SELECT * 
			      FROM FINANCIALS.MARVELL_DEMO.ITEM_DETAILS  ITD
			      WHERE ITEM_STAGE  = 'FG'
                              ORDER BY ITEM_WID  LIMIT 10;''',
            }, 
            {
                "input": "what is average actual invetory per quarter per type",
                "sql_cmd": '''SELECT QUARTER_NAME AS "QUARTER NAME", TYPE AS "TYPE", to_varchar(avg(AMOUNT), '$ 999,999,999.90') AS "AVERAGE AMOUNT"
                           FROM FINANCIALS.MARVELL_DEMO.INVENTORY_ACTUALS GROUP BY QUARTER_NAME, TYPE''',
            },
            {
                "input": "what is maximum and minimum projected inventory amount per quarter per type",
                "sql_cmd": '''SELECT QUARTER_NAME AS "QUARTER NAME", TYPE AS "TYPE", to_varchar(max(AMOUNT), '$ 999,999,999.90') AS "MAX AMOUNT",
                           to_varchar(min(AMOUNT), '$ 999,999,999.90') AS "MIN AMOUNT" FROM FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY GROUP BY QUARTER_NAME, TYPE;''',
            },
            {
                "input": "what is inventory on hand for next 2 quarters for the business unit Switch.",
                "sql_cmd": '''SELECT A.QUARTER_NAME AS "QUARTER NAME", B.BU AS "BU", to_varchar(A.AMOUNT, '$ 999,999,999.90') AS "ON-HAND INVENTORY AMOUNT"
                            FROM FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS A LEFT JOIN FINANCIALS.MARVELL_DEMO.ITEM_DETAILS B ON B.ITEM_WID = A.ITEM_WID WHERE A.QUARTER_NAME IN ('2025-Q1','2025-Q2');''',
            },
            {
                "input": "what is inventory on hand for previous 2 quarters for the business unit Switch.",
                "sql_cmd": '''SELECT A.QUARTER_NAME AS "QUARTER NAME", B.BU AS "BU", to_varchar(A.AMOUNT, '$ 999,999,999.90') AS "ON-HAND INVENTORY AMOUNT"
                            FROM FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS A LEFT JOIN FINANCIALS.MARVELL_DEMO.ITEM_DETAILS B ON B.ITEM_WID = A.ITEM_WID WHERE A.QUARTER_NAME IN ('2024-Q3','2024-Q2');''',
            },            {
                "input": "what is inventory on hand for current quarter for the business unit Switch.",
                "sql_cmd": '''SELECT A.QUARTER_NAME AS "QUARTER NAME", B.BU AS "BU", to_varchar(A.AMOUNT, '$ 999,999,999.90') AS "ON-HAND INVENTORY AMOUNT"
                            FROM FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS A LEFT JOIN FINANCIALS.MARVELL_DEMO.ITEM_DETAILS B ON B.ITEM_WID = A.ITEM_WID WHERE A.QUARTER_NAME = '2024-Q4';''',
            },
        ]
        return examples

    @staticmethod
    def get_example_template():
        template = """
        Input: {input}
        SQL_cmd: {sql_cmd}\n
        """
        example_variables = ["input", "sql_cmd"]
        return template, example_variables
