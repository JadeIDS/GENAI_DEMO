create or replace TABLE FINANCIALS.MARVELL_DEMO.INVENTORY_ACTUALS (
	ITEM_WID NUMBER(38,0),
	AMOUNT FLOAT,
	QUARTER_NAME VARCHAR(16777216),
	TYPE VARCHAR(16777216)
);

create or replace TABLE FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS (
	ITEM_WID NUMBER(38,0) ,
	AMOUNT FLOAT,
	QUANTITY FLOAT,
	QUARTER_NAME VARCHAR(16777216),
	CUM_YIELD FLOAT,
	YIELD_QUANTITY FLOAT
);


create or replace TABLE FINANCIALS.MARVELL_DEMO.ITEM_DETAILS (
	ITEM_WID NUMBER(38,0),
	COMPANY VARCHAR(16777216),
	DIVISION VARCHAR(16777216),
	BU VARCHAR(16777216),
	PRODUCT_LINE VARCHAR(16777216),
	SUB_PRODUCT_LINE VARCHAR(16777216),
	FINANCIAL_GROUP VARCHAR(16777216),
	PART_NUMBER VARCHAR(16777216),
	ITEM_STAGE VARCHAR(16777216),
	ASSY_PART_NUMBER VARCHAR(16777216),
	DIE_PART_NUMBER VARCHAR(16777216),
	ECCN FLOAT,
	FAB_PART_NUMBER VARCHAR(16777216),
	SCHEDULE_B_NUM FLOAT,
	TEST_PART_NUMBER VARCHAR(16777216),
	PLATFORM FLOAT,
	BASE_DIE VARCHAR(16777216),
	BASE_DIE_REV VARCHAR(16777216),
	FAB_HOUSE VARCHAR(16777216),
	DESIGN_COUNTRY FLOAT,
	MARKETING_ITEM VARCHAR(16777216),
	ITEM_TYPE VARCHAR(16777216),
	PLANNER_CODE FLOAT,
	INVENTORY_ORGANIZATION NUMBER(38,0),
	COUNTRY_OF_MANUFACTURING FLOAT,
	DEVICE VARCHAR(16777216),
	PRODUCT_GROUP VARCHAR(16777216),
	PROJECT_CODE FLOAT,
	PROJECT_CODE_DESCRIPTION VARCHAR(16777216),
	PACKAGE_CODE VARCHAR(16777216),
	SERVICE_PART_NUMBER VARCHAR(16777216),
	LICENSE_EXCEPTION FLOAT,
	END_MARKET VARCHAR(16777216),
	SUB_MARKET VARCHAR(16777216),
	MODEL_NUMBER VARCHAR(16777216),
	SPEED FLOAT,
	ITEM_TYPE_GROUP VARCHAR(16777216),
	MPR_YEAR_CATEGORY FLOAT,
	MPR_YEAR_CATEGORY_GROUP VARCHAR(16777216),
	MPR_YEAR_CATEGORY_SORTING NUMBER(38,0),
	CURRENT_MODEL_NUMBER VARCHAR(16777216),
	SALES_COMP_DIVISION VARCHAR(16777216),
	SALES_COMP_BU VARCHAR(16777216),
	SALES_COMP_PRODUCT_LINE VARCHAR(16777216),
	SALES_COMP_FINANCIAL_GROUP VARCHAR(16777216),
	SALES_COMP_SUB_PRODUCT_LINE VARCHAR(16777216),
	TECHNOLOGY_GROUP VARCHAR(16777216),
	IDENTIFIED_IP FLOAT,
	BUSINESS_OWNER VARCHAR(16777216),
	PRODUCT_LINE_MANAGER FLOAT
);


create or replace TABLE FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY (
	ITEM_WID NUMBER(38,0),
	AMOUNT FLOAT,
	QUARTER_NAME VARCHAR(16777216),
	TYPE VARCHAR(16777216)
);


**Table 1: FINANCIALS.MARVELL_DEMO.INVENTORY_ACTUALS (Stores Actual Inventory ITEM information for  Different Type of Items)
 
- ITEM_WID		-  Unique ID for an ITEM refers to ITEM_WID in table FINANCIALS.MARVELL_DEMO.ITEM_DETAILS
- AMOUNT		-  Amount associated with the ITEM 
- QUARTER_NAME	-  Quarter when Item inventory	was populated
- TYPE		    -  Type of Inventory for Actual Items


**Table 2: FINANCIALS.MARVELL_DEMO.INVENTORY_ON_HANDS (Stores On Hand Inventory information for an ITEM for Different Quarters)
 
- ITEM_WID			-  Unique ID for an ITEM refers to ITEM_WID in table FINANCIALS.MARVELL_DEMO.ITEM_DETAILS
- AMOUNT			-  Amount associated with the ITEM	
- QUANTITY			-  Number of ITEM Quantities Per Quarter
- QUARTER_NAME		-  Quarter when Item inventory	was populated
- CUM_YIELD			-  Cummulative Yield for an Item per Quarter
- YIELD_QUANTITY 	-  Quantity Yield for an Item per Quarter

**Table 3: FINANCIALS.MARVELL_DEMO.ITEM_DETAILS (This is the ITEM master table that stores ITEM Information including their Division,BU, ProductLine ETC.)


- ITEM_WID 						-	Unique ID for an ITEM	
- COMPANY                       -	NAME of the company Item belongs to 
- DIVISION                      -	Division In the company Item belongs to 
- BU                            -	BU stands for business unit and it means BU in the company Item belongs to
- PRODUCT_LINE                  -	Product Line In the company Item belongs to
- SUB_PRODUCT_LINE              -	Sub Product Line In the company Item belongs to
- FINANCIAL_GROUP               -	Financial Group In the company Item belongs to
- PART_NUMBER                   -	Part Number for an ITEM
- ITEM_STAGE                    -	Stage in which Item is currently in Product Life Cycle. 
- ASSY_PART_NUMBER              -	Number when the part of Item is in Assembly stage.
- DIE_PART_NUMBER               -	Number when the part of Item is in Die stage.
- ECCN                          -	ECCN Number for an Item to determine licencing information.
- FAB_PART_NUMBER               -	Number when the part of Item is in FAB stage.
- SCHEDULE_B_NUM                -	Specific Classification Code for Exporting Goods.
- TEST_PART_NUMBER              -	Number when the part of Item is in Test stage.
- PLATFORM                      -	
- BASE_DIE                      -	Id for the Die.
- BASE_DIE_REV                  -	Revised Id for the Die.
- FAB_HOUSE                     -	Fabrication House Name for the Item.
- DESIGN_COUNTRY                -	Country Name where the Item is Designed.
- MARKETING_ITEM                -	Item Number Associated for  Marketting Reference.
- ITEM_TYPE                     -	Type of Item 
- PLANNER_CODE                  -	PLanner Code for an Item which are up to date with Factory Status.
- INVENTORY_ORGANIZATION        -	Unique Number of Inventory Organization
- COUNTRY_OF_MANUFACTURING      -	Country where Item is Manufactured.
- DEVICE                        -	Device Name of the Item.
- PRODUCT_GROUP                 -	Type of Product for Eg if its Wfr or IC
- PROJECT_CODE                  -	Project Code Associated with an Item.
- PROJECT_CODE_DESCRIPTION      -	Description of a Project Code.
- PACKAGE_CODE                  -	Package Code for an Item.
- SERVICE_PART_NUMBER           -	Service Part Number for an Item.
- LICENSE_EXCEPTION             -	Export Licence Exception Number for the Item.
- END_MARKET                    -	Describes which category of Market Item belongs to during Sale.  
- SUB_MARKET 					-	Describes which Sub category of Market Item belongs to during Sale.  
- MODEL_NUMBER       			-	Model Number for an ITEM. 
- SPEED                         -	Speed for an Item from performance Characteristics.
- ITEM_TYPE_GROUP               -	Item Group Type for an Item Listed
- MPR_YEAR_CATEGORY             -	MPR_YEAR_CATEGORY for an Item.
- MPR_YEAR_CATEGORY_GROUP       -	Categroy Group for MPR
- MPR_YEAR_CATEGORY_SORTING     -	
- CURRENT_MODEL_NUMBER          -	Current Model Number for an ITEM.
- SALES_COMP_DIVISION           -	Division for the Sales Team doing the Sales for an ITEM.
- SALES_COMP_BU                 -	BU for the Sales Team doing the Sales for an ITEM.
- SALES_COMP_PRODUCT_LINE       -	Product Line for the Sales Team doing the Sales for an ITEM.
- SALES_COMP_FINANCIAL_GROUP    -	Financial Group for the Sales Team doing the Sales for an ITEM.
- SALES_COMP_SUB_PRODUCT_LINE   -	Sub Product Line for the Sales Team doing the Sales for an ITEM.
- TECHNOLOGY_GROUP              -	Tech Group for an ITEM.
- IDENTIFIED_IP                 -	Intellectual property for an Item.
- BUSINESS_OWNER                -	Busines owner of the ITEM.
- PRODUCT_LINE_MANAGER          -	Line Manager for an ITEM.


**Table 4: FINANCIALS.MARVELL_DEMO.PROJECTED_INVENTORY (Stores Projected ITEM Information in the inventory )


- ITEM_WID						-   Unique ID for an ITEM refers to ITEM_WID in table FINANCIALS.MARVELL_DEMO.ITEM_DETAILS
- AMOUNT						-	Amount associated with the ITEM	
- QUARTER_NAME					-   Quarter when Item inventory	was populated
- TYPE 						    -   Type of Inventory for Projected Items
