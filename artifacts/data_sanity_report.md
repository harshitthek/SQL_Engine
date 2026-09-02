# Data Sanity Inspection Report (20 Random Samples)
This report details 20 randomly sampled examples across various databases and complexity tiers,verifying schema correctness, question clarity, foreign key linkages, and target SQL.

---

## Sample 1/20: `music_2` (Complexity: **hard**)
- **Token Count**: 434 tokens- **Sanity Checks**: PASSED
**Question**: Who performed the song named "Le Pop"?
**Target SQL**:
```sql
SELECT T2.firstname ,  T2.lastname FROM Performance AS T1 JOIN Band AS T2 ON T1.bandmate  =  T2.id JOIN Songs AS T3 ON T3.SongId  =  T1.SongId WHERE T3.Title  =  "Le Pop"
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Songs` (
  `SongId` NUMBER PRIMARY KEY,
  `Title` TEXT
);

CREATE TABLE `Albums` (
  `AId` NUMBER PRIMARY KEY,
  `Title` TEXT,
  `Year` NUMBER,
  `Label` TEXT,
  `Type` TEXT
);

CREATE TABLE `Band` (
  `Id` NUMBER PRIMARY KEY,
  `Firstname` TEXT,
  `Lastname` TEXT
);

CREATE TABLE `Instruments` (
  `SongId` NUMBER PRIMARY KEY,
  `BandmateId` NUMBER,
  `Instrument` TEXT,
  FOREIGN KEY (`BandmateId`) REFERENCES `Band`(`Id`),
  FOREIGN KEY (`SongId`) REFERENCES `Songs`(`SongId`)
);

CREATE TABLE `Performance` (
  `SongId` NUMBER PRIMARY KEY,
  `Bandmate` NUMBER,
  `StagePosition` TEXT,
  FOREIGN KEY (`Bandmate`) REFERENCES `Band`(`Id`),
  FOREIGN KEY (`SongId`) REFERENCES `Songs`(`SongId`)
);

CREATE TABLE `Tracklists` (
  `AlbumId` NUMBER PRIMARY KEY,
  `Position` NUMBER,
  `SongId` NUMBER,
  FOREIGN KEY (`AlbumId`) REFERENCES `Albums`(`AId`),
  FOREIGN KEY (`SongId`) REFERENCES `Songs`(`SongId`)
);

CREATE TABLE `Vocals` (
  `SongId` NUMBER PRIMARY KEY,
  `Bandmate` NUMBER,
  `Type` TEXT,
  FOREIGN KEY (`Bandmate`) REFERENCES `Band`(`Id`),
  FOREIGN KEY (`SongId`) REFERENCES `Songs`(`SongId`)
);

### Question:
Who performed the song named "Le Pop"?

### SQL:

```
</details>

---

## Sample 2/20: `insurance_fnol` (Complexity: **hard**)
- **Token Count**: 470 tokens- **Sanity Checks**: PASSED
**Question**: Tell me the types of the policy used by the customer named "Dayana Robel".
**Target SQL**:
```sql
SELECT DISTINCT t3.policy_type_code FROM customers AS t1 JOIN customers_policies AS t2 ON t1.customer_id  =  t2.customer_id JOIN available_policies AS t3 ON t2.policy_id  =  t3.policy_id WHERE t1.customer_name  =  "Dayana Robel"
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Customers` (
  `Customer_ID` NUMBER PRIMARY KEY,
  `Customer_name` TEXT
);

CREATE TABLE `Services` (
  `Service_ID` NUMBER PRIMARY KEY,
  `Service_name` TEXT
);

CREATE TABLE `Available_Policies` (
  `Policy_ID` NUMBER PRIMARY KEY,
  `policy_type_code` TEXT,
  `Customer_Phone` TEXT
);

CREATE TABLE `Customers_Policies` (
  `Customer_ID` NUMBER PRIMARY KEY,
  `Policy_ID` NUMBER,
  `Date_Opened` TIME,
  `Date_Closed` TIME,
  FOREIGN KEY (`Policy_ID`) REFERENCES `Available_Policies`(`Policy_ID`),
  FOREIGN KEY (`Customer_ID`) REFERENCES `Customers`(`Customer_ID`)
);

CREATE TABLE `First_Notification_of_Loss` (
  `FNOL_ID` NUMBER PRIMARY KEY,
  `Customer_ID` NUMBER,
  `Policy_ID` NUMBER,
  `Service_ID` NUMBER,
  FOREIGN KEY (`Customer_ID`) REFERENCES `Customers_Policies`(`Customer_ID`),
  FOREIGN KEY (`Policy_ID`) REFERENCES `Customers_Policies`(`Policy_ID`),
  FOREIGN KEY (`Service_ID`) REFERENCES `Services`(`Service_ID`)
);

CREATE TABLE `Claims` (
  `Claim_ID` NUMBER PRIMARY KEY,
  `FNOL_ID` NUMBER,
  `Effective_Date` TIME,
  FOREIGN KEY (`FNOL_ID`) REFERENCES `First_Notification_of_Loss`(`FNOL_ID`)
);

CREATE TABLE `Settlements` (
  `Settlement_ID` NUMBER PRIMARY KEY,
  `Claim_ID` NUMBER,
  `Effective_Date` TIME,
  `Settlement_Amount` NUMBER,
  FOREIGN KEY (`Claim_ID`) REFERENCES `Claims`(`Claim_ID`)
);

### Question:
Tell me the types of the policy used by the customer named "Dayana Robel".

### SQL:

```
</details>

---

## Sample 3/20: `bike_1` (Complexity: **medium**)
- **Token Count**: 508 tokens- **Sanity Checks**: PASSED
**Question**: What are the different ids and names of the stations that have had more than 12 bikes available?
**Target SQL**:
```sql
SELECT DISTINCT T1.id ,  T1.name FROM station AS T1 JOIN status AS T2 ON T1.id  =  T2.station_id WHERE T2.bikes_available  >  12
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `station` (
  `id` NUMBER PRIMARY KEY,
  `name` TEXT,
  `lat` NUMBER,
  `long` NUMBER,
  `dock_count` NUMBER,
  `city` TEXT,
  `installation_date` TEXT
);

CREATE TABLE `status` (
  `station_id` NUMBER,
  `bikes_available` NUMBER,
  `docks_available` NUMBER,
  `time` TEXT,
  FOREIGN KEY (`station_id`) REFERENCES `station`(`id`)
);

CREATE TABLE `trip` (
  `id` NUMBER PRIMARY KEY,
  `duration` NUMBER,
  `start_date` TEXT,
  `start_station_name` TEXT,
  `start_station_id` NUMBER,
  `end_date` TEXT,
  `end_station_name` TEXT,
  `end_station_id` NUMBER,
  `bike_id` NUMBER,
  `subscription_type` TEXT,
  `zip_code` NUMBER
);

CREATE TABLE `weather` (
  `date` TEXT,
  `max_temperature_f` NUMBER,
  `mean_temperature_f` NUMBER,
  `min_temperature_f` NUMBER,
  `max_dew_point_f` NUMBER,
  `mean_dew_point_f` NUMBER,
  `min_dew_point_f` NUMBER,
  `max_humidity` NUMBER,
  `mean_humidity` NUMBER,
  `min_humidity` NUMBER,
  `max_sea_level_pressure_inches` NUMBER,
  `mean_sea_level_pressure_inches` NUMBER,
  `min_sea_level_pressure_inches` NUMBER,
  `max_visibility_miles` NUMBER,
  `mean_visibility_miles` NUMBER,
  `min_visibility_miles` NUMBER,
  `max_wind_Speed_mph` NUMBER,
  `mean_wind_speed_mph` NUMBER,
  `max_gust_speed_mph` NUMBER,
  `precipitation_inches` NUMBER,
  `cloud_cover` NUMBER,
  `events` TEXT,
  `wind_dir_degrees` NUMBER,
  `zip_code` NUMBER
);

### Question:
What are the different ids and names of the stations that have had more than 12 bikes available?

### SQL:

```
</details>

---

## Sample 4/20: `customers_and_addresses` (Complexity: **medium**)
- **Token Count**: 450 tokens- **Sanity Checks**: PASSED
**Question**: Which customer's name contains "Alex"? Find the full name.
**Target SQL**:
```sql
SELECT customer_name FROM customers WHERE customer_name LIKE "%Alex%"
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Addresses` (
  `address_id` NUMBER PRIMARY KEY,
  `address_content` TEXT,
  `city` TEXT,
  `zip_postcode` TEXT,
  `state_province_county` TEXT,
  `country` TEXT,
  `other_address_details` TEXT
);

CREATE TABLE `Products` (
  `product_id` NUMBER PRIMARY KEY,
  `product_details` TEXT
);

CREATE TABLE `Customers` (
  `customer_id` NUMBER PRIMARY KEY,
  `payment_method` TEXT,
  `customer_name` TEXT,
  `date_became_customer` TIME,
  `other_customer_details` TEXT
);

CREATE TABLE `Customer_Addresses` (
  `customer_id` NUMBER,
  `address_id` NUMBER,
  `date_address_from` TIME,
  `address_type` TEXT,
  `date_address_to` TIME,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`),
  FOREIGN KEY (`address_id`) REFERENCES `Addresses`(`address_id`)
);

CREATE TABLE `Customer_Contact_Channels` (
  `customer_id` NUMBER,
  `channel_code` TEXT,
  `active_from_date` TIME,
  `active_to_date` TIME,
  `contact_number` TEXT,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`)
);

CREATE TABLE `Customer_Orders` (
  `order_id` NUMBER PRIMARY KEY,
  `customer_id` NUMBER,
  `order_status` TEXT,
  `order_date` TIME,
  `order_details` TEXT,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`)
);

CREATE TABLE `Order_Items` (
  `order_id` NUMBER,
  `product_id` NUMBER,
  `order_quantity` TEXT,
  FOREIGN KEY (`order_id`) REFERENCES `Customer_Orders`(`order_id`),
  FOREIGN KEY (`product_id`) REFERENCES `Products`(`product_id`)
);

### Question:
Which customer's name contains "Alex"? Find the full name.

### SQL:

```
</details>

---

## Sample 5/20: `machine_repair` (Complexity: **hard**)
- **Token Count**: 320 tokens- **Sanity Checks**: PASSED
**Question**: Show names of technicians in ascending order of quality rank of the machine they are assigned.
**Target SQL**:
```sql
SELECT T3.Name FROM repair_assignment AS T1 JOIN machine AS T2 ON T1.machine_id  =  T2.machine_id JOIN technician AS T3 ON T1.technician_ID  =  T3.technician_ID ORDER BY T2.quality_rank
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `repair` (
  `repair_ID` NUMBER PRIMARY KEY,
  `name` TEXT,
  `Launch_Date` TEXT,
  `Notes` TEXT
);

CREATE TABLE `machine` (
  `Machine_ID` NUMBER PRIMARY KEY,
  `Making_Year` NUMBER,
  `Class` TEXT,
  `Team` TEXT,
  `Machine_series` TEXT,
  `value_points` NUMBER,
  `quality_rank` NUMBER
);

CREATE TABLE `technician` (
  `technician_id` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Team` TEXT,
  `Starting_Year` NUMBER,
  `Age` NUMBER
);

CREATE TABLE `repair_assignment` (
  `technician_id` NUMBER PRIMARY KEY,
  `repair_ID` NUMBER,
  `Machine_ID` NUMBER,
  FOREIGN KEY (`Machine_ID`) REFERENCES `machine`(`Machine_ID`),
  FOREIGN KEY (`repair_ID`) REFERENCES `repair`(`repair_ID`),
  FOREIGN KEY (`technician_id`) REFERENCES `technician`(`technician_id`)
);

### Question:
Show names of technicians in ascending order of quality rank of the machine they are assigned.

### SQL:

```
</details>

---

## Sample 6/20: `gas_company` (Complexity: **medium**)
- **Token Count**: 257 tokens- **Sanity Checks**: PASSED
**Question**: For each headquarter, what are the headquarter and how many companies are centered there?
**Target SQL**:
```sql
SELECT headquarters ,  count(*) FROM company GROUP BY headquarters
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `company` (
  `Company_ID` NUMBER PRIMARY KEY,
  `Rank` NUMBER,
  `Company` TEXT,
  `Headquarters` TEXT,
  `Main_Industry` TEXT,
  `Sales_billion` NUMBER,
  `Profits_billion` NUMBER,
  `Assets_billion` NUMBER,
  `Market_Value` NUMBER
);

CREATE TABLE `gas_station` (
  `Station_ID` NUMBER PRIMARY KEY,
  `Open_Year` NUMBER,
  `Location` TEXT,
  `Manager_Name` TEXT,
  `Vice_Manager_Name` TEXT,
  `Representative_Name` TEXT
);

CREATE TABLE `station_company` (
  `Station_ID` NUMBER PRIMARY KEY,
  `Company_ID` NUMBER,
  `Rank_of_the_Year` NUMBER,
  FOREIGN KEY (`Company_ID`) REFERENCES `company`(`Company_ID`),
  FOREIGN KEY (`Station_ID`) REFERENCES `gas_station`(`Station_ID`)
);

### Question:
For each headquarter, what are the headquarter and how many companies are centered there?

### SQL:

```
</details>

---

## Sample 7/20: `browser_web` (Complexity: **medium**)
- **Token Count**: 195 tokens- **Sanity Checks**: PASSED
**Question**: List the ids, names and market shares of all browsers.
**Target SQL**:
```sql
SELECT id ,  name ,  market_share FROM browser
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Web_client_accelerator` (
  `id` NUMBER PRIMARY KEY,
  `name` TEXT,
  `Operating_system` TEXT,
  `Client` TEXT,
  `Connection` TEXT
);

CREATE TABLE `browser` (
  `id` NUMBER PRIMARY KEY,
  `name` TEXT,
  `market_share` NUMBER
);

CREATE TABLE `accelerator_compatible_browser` (
  `accelerator_id` NUMBER PRIMARY KEY,
  `browser_id` NUMBER,
  `compatible_since_year` NUMBER,
  FOREIGN KEY (`browser_id`) REFERENCES `browser`(`id`),
  FOREIGN KEY (`accelerator_id`) REFERENCES `Web_client_accelerator`(`id`)
);

### Question:
List the ids, names and market shares of all browsers.

### SQL:

```
</details>

---

## Sample 8/20: `climbing` (Complexity: **hard**)
- **Token Count**: 180 tokens- **Sanity Checks**: PASSED
**Question**: What are the names of countains that no climber has climbed?
**Target SQL**:
```sql
SELECT Name FROM mountain WHERE Mountain_ID NOT IN (SELECT Mountain_ID FROM climber)
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `mountain` (
  `Mountain_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Height` NUMBER,
  `Prominence` NUMBER,
  `Range` TEXT,
  `Country` TEXT
);

CREATE TABLE `climber` (
  `Climber_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Country` TEXT,
  `Time` TEXT,
  `Points` NUMBER,
  `Mountain_ID` NUMBER,
  FOREIGN KEY (`Mountain_ID`) REFERENCES `mountain`(`Mountain_ID`)
);

### Question:
What are the names of countains that no climber has climbed?

### SQL:

```
</details>

---

## Sample 9/20: `game_1` (Complexity: **simple**)
- **Token Count**: 277 tokens- **Sanity Checks**: PASSED
**Question**: How many students play video games?
**Target SQL**:
```sql
SELECT count(DISTINCT StuID) FROM Plays_games
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Student` (
  `StuID` NUMBER PRIMARY KEY,
  `LName` TEXT,
  `Fname` TEXT,
  `Age` NUMBER,
  `Sex` TEXT,
  `Major` NUMBER,
  `Advisor` NUMBER,
  `city_code` TEXT
);

CREATE TABLE `Video_Games` (
  `GameID` NUMBER PRIMARY KEY,
  `GName` TEXT,
  `GType` TEXT
);

CREATE TABLE `Plays_Games` (
  `StuID` NUMBER,
  `GameID` NUMBER,
  `Hours_Played` NUMBER,
  FOREIGN KEY (`StuID`) REFERENCES `Student`(`StuID`),
  FOREIGN KEY (`GameID`) REFERENCES `Video_Games`(`GameID`)
);

CREATE TABLE `SportsInfo` (
  `StuID` NUMBER,
  `SportName` TEXT,
  `HoursPerWeek` NUMBER,
  `GamesPlayed` NUMBER,
  `OnScholarship` TEXT,
  FOREIGN KEY (`StuID`) REFERENCES `Student`(`StuID`)
);

### Question:
How many students play video games?

### SQL:

```
</details>

---

## Sample 10/20: `chinook_1` (Complexity: **hard**)
- **Token Count**: 789 tokens- **Sanity Checks**: PASSED
**Question**: Show the album names and ids for albums that contain tracks with unit price bigger than 1.
**Target SQL**:
```sql
SELECT T1.Title ,  T2.AlbumID FROM ALBUM AS T1 JOIN TRACK AS T2 ON T1.AlbumId  =  T2.AlbumId WHERE T2.UnitPrice  >  1 GROUP BY T2.AlbumID
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Album` (
  `AlbumId` NUMBER PRIMARY KEY,
  `Title` TEXT,
  `ArtistId` NUMBER,
  FOREIGN KEY (`ArtistId`) REFERENCES `Artist`(`ArtistId`)
);

CREATE TABLE `Artist` (
  `ArtistId` NUMBER PRIMARY KEY,
  `Name` TEXT
);

CREATE TABLE `Customer` (
  `CustomerId` NUMBER PRIMARY KEY,
  `FirstName` TEXT,
  `LastName` TEXT,
  `Company` TEXT,
  `Address` TEXT,
  `City` TEXT,
  `State` TEXT,
  `Country` TEXT,
  `PostalCode` TEXT,
  `Phone` TEXT,
  `Fax` TEXT,
  `Email` TEXT,
  `SupportRepId` NUMBER,
  FOREIGN KEY (`SupportRepId`) REFERENCES `Employee`(`EmployeeId`)
);

CREATE TABLE `Employee` (
  `EmployeeId` NUMBER PRIMARY KEY,
  `LastName` TEXT,
  `FirstName` TEXT,
  `Title` TEXT,
  `ReportsTo` NUMBER,
  `BirthDate` TIME,
  `HireDate` TIME,
  `Address` TEXT,
  `City` TEXT,
  `State` TEXT,
  `Country` TEXT,
  `PostalCode` TEXT,
  `Phone` TEXT,
  `Fax` TEXT,
  `Email` TEXT,
  FOREIGN KEY (`ReportsTo`) REFERENCES `Employee`(`EmployeeId`)
);

CREATE TABLE `Genre` (
  `GenreId` NUMBER PRIMARY KEY,
  `Name` TEXT
);

CREATE TABLE `Invoice` (
  `InvoiceId` NUMBER PRIMARY KEY,
  `CustomerId` NUMBER,
  `InvoiceDate` TIME,
  `BillingAddress` TEXT,
  `BillingCity` TEXT,
  `BillingState` TEXT,
  `BillingCountry` TEXT,
  `BillingPostalCode` TEXT,
  `Total` NUMBER,
  FOREIGN KEY (`CustomerId`) REFERENCES `Customer`(`CustomerId`)
);

CREATE TABLE `InvoiceLine` (
  `InvoiceLineId` NUMBER PRIMARY KEY,
  `InvoiceId` NUMBER,
  `TrackId` NUMBER,
  `UnitPrice` NUMBER,
  `Quantity` NUMBER,
  FOREIGN KEY (`TrackId`) REFERENCES `Track`(`TrackId`),
  FOREIGN KEY (`InvoiceId`) REFERENCES `Invoice`(`InvoiceId`)
);

CREATE TABLE `MediaType` (
  `MediaTypeId` NUMBER PRIMARY KEY,
  `Name` TEXT
);

CREATE TABLE `Playlist` (
  `PlaylistId` NUMBER PRIMARY KEY,
  `Name` TEXT
);

CREATE TABLE `PlaylistTrack` (
  `PlaylistId` NUMBER PRIMARY KEY,
  `TrackId` NUMBER,
  FOREIGN KEY (`TrackId`) REFERENCES `Track`(`TrackId`),
  FOREIGN KEY (`PlaylistId`) REFERENCES `Playlist`(`PlaylistId`)
);

CREATE TABLE `Track` (
  `TrackId` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `AlbumId` NUMBER,
  `MediaTypeId` NUMBER,
  `GenreId` NUMBER,
  `Composer` TEXT,
  `Milliseconds` NUMBER,
  `Bytes` NUMBER,
  `UnitPrice` NUMBER,
  FOREIGN KEY (`MediaTypeId`) REFERENCES `MediaType`(`MediaTypeId`),
  FOREIGN KEY (`GenreId`) REFERENCES `Genre`(`GenreId`),
  FOREIGN KEY (`AlbumId`) REFERENCES `Album`(`AlbumId`)
);

### Question:
Show the album names and ids for albums that contain tracks with unit price bigger than 1.

### SQL:

```
</details>

---

## Sample 11/20: `products_gen_characteristics` (Complexity: **hard**)
- **Token Count**: 438 tokens- **Sanity Checks**: PASSED
**Question**: What are the descriptions of the categories that products with product descriptions that contain the letter t are in?
**Target SQL**:
```sql
SELECT T1.product_category_description FROM ref_product_categories AS T1 JOIN products AS T2 ON T1.product_category_code  =  T2.product_category_code WHERE T2.product_description LIKE '%t%'
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Ref_Characteristic_Types` (
  `characteristic_type_code` TEXT PRIMARY KEY,
  `characteristic_type_description` TEXT
);

CREATE TABLE `Ref_Colors` (
  `color_code` TEXT PRIMARY KEY,
  `color_description` TEXT
);

CREATE TABLE `Ref_Product_Categories` (
  `product_category_code` TEXT PRIMARY KEY,
  `product_category_description` TEXT,
  `unit_of_measure` TEXT
);

CREATE TABLE `Characteristics` (
  `characteristic_id` NUMBER PRIMARY KEY,
  `characteristic_type_code` TEXT,
  `characteristic_data_type` TEXT,
  `characteristic_name` TEXT,
  `other_characteristic_details` TEXT,
  FOREIGN KEY (`characteristic_type_code`) REFERENCES `Ref_Characteristic_Types`(`characteristic_type_code`)
);

CREATE TABLE `Products` (
  `product_id` NUMBER PRIMARY KEY,
  `color_code` TEXT,
  `product_category_code` TEXT,
  `product_name` TEXT,
  `typical_buying_price` TEXT,
  `typical_selling_price` TEXT,
  `product_description` TEXT,
  `other_product_details` TEXT,
  FOREIGN KEY (`color_code`) REFERENCES `Ref_Colors`(`color_code`),
  FOREIGN KEY (`product_category_code`) REFERENCES `Ref_Product_Categories`(`product_category_code`)
);

CREATE TABLE `Product_Characteristics` (
  `product_id` NUMBER,
  `characteristic_id` NUMBER,
  `product_characteristic_value` TEXT,
  FOREIGN KEY (`product_id`) REFERENCES `Products`(`product_id`),
  FOREIGN KEY (`characteristic_id`) REFERENCES `Characteristics`(`characteristic_id`)
);

### Question:
What are the descriptions of the categories that products with product descriptions that contain the letter t are in?

### SQL:

```
</details>

---

## Sample 12/20: `customers_and_addresses` (Complexity: **hard**)
- **Token Count**: 468 tokens- **Sanity Checks**: PASSED
**Question**: What are the names of customers using the most popular payment method?
**Target SQL**:
```sql
SELECT customer_name FROM customers WHERE payment_method  =  (SELECT payment_method FROM customers GROUP BY payment_method ORDER BY count(*) DESC LIMIT 1)
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Addresses` (
  `address_id` NUMBER PRIMARY KEY,
  `address_content` TEXT,
  `city` TEXT,
  `zip_postcode` TEXT,
  `state_province_county` TEXT,
  `country` TEXT,
  `other_address_details` TEXT
);

CREATE TABLE `Products` (
  `product_id` NUMBER PRIMARY KEY,
  `product_details` TEXT
);

CREATE TABLE `Customers` (
  `customer_id` NUMBER PRIMARY KEY,
  `payment_method` TEXT,
  `customer_name` TEXT,
  `date_became_customer` TIME,
  `other_customer_details` TEXT
);

CREATE TABLE `Customer_Addresses` (
  `customer_id` NUMBER,
  `address_id` NUMBER,
  `date_address_from` TIME,
  `address_type` TEXT,
  `date_address_to` TIME,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`),
  FOREIGN KEY (`address_id`) REFERENCES `Addresses`(`address_id`)
);

CREATE TABLE `Customer_Contact_Channels` (
  `customer_id` NUMBER,
  `channel_code` TEXT,
  `active_from_date` TIME,
  `active_to_date` TIME,
  `contact_number` TEXT,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`)
);

CREATE TABLE `Customer_Orders` (
  `order_id` NUMBER PRIMARY KEY,
  `customer_id` NUMBER,
  `order_status` TEXT,
  `order_date` TIME,
  `order_details` TEXT,
  FOREIGN KEY (`customer_id`) REFERENCES `Customers`(`customer_id`)
);

CREATE TABLE `Order_Items` (
  `order_id` NUMBER,
  `product_id` NUMBER,
  `order_quantity` TEXT,
  FOREIGN KEY (`order_id`) REFERENCES `Customer_Orders`(`order_id`),
  FOREIGN KEY (`product_id`) REFERENCES `Products`(`product_id`)
);

### Question:
What are the names of customers using the most popular payment method?

### SQL:

```
</details>

---

## Sample 13/20: `network_2` (Complexity: **extra-hard**)
- **Token Count**: 198 tokens- **Sanity Checks**: PASSED
**Question**: Whare the names, friends, and ages of all people who are older than the average age of a person?
**Target SQL**:
```sql
SELECT DISTINCT T2.name ,  T2.friend ,  T1.age FROM Person AS T1 JOIN PersonFriend AS T2 ON T1.name  =  T2.friend WHERE T1.age  >  (SELECT avg(age) FROM person)
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Person` (
  `name` TEXT PRIMARY KEY,
  `age` NUMBER,
  `city` TEXT,
  `gender` TEXT,
  `job` TEXT
);

CREATE TABLE `PersonFriend` (
  `name` TEXT,
  `friend` TEXT,
  `year` NUMBER,
  FOREIGN KEY (`friend`) REFERENCES `Person`(`name`),
  FOREIGN KEY (`name`) REFERENCES `Person`(`name`)
);

### Question:
Whare the names, friends, and ages of all people who are older than the average age of a person?

### SQL:

```
</details>

---

## Sample 14/20: `customers_card_transactions` (Complexity: **medium**)
- **Token Count**: 332 tokens- **Sanity Checks**: PASSED
**Question**: What are the different customer ids, and how many cards does each one hold?
**Target SQL**:
```sql
SELECT customer_id ,  count(*) FROM Customers_cards GROUP BY customer_id
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `Accounts` (
  `account_id` NUMBER PRIMARY KEY,
  `customer_id` NUMBER,
  `account_name` TEXT,
  `other_account_details` TEXT
);

CREATE TABLE `Customers` (
  `customer_id` NUMBER PRIMARY KEY,
  `customer_first_name` TEXT,
  `customer_last_name` TEXT,
  `customer_address` TEXT,
  `customer_phone` TEXT,
  `customer_email` TEXT,
  `other_customer_details` TEXT
);

CREATE TABLE `Customers_Cards` (
  `card_id` NUMBER PRIMARY KEY,
  `customer_id` NUMBER,
  `card_type_code` TEXT,
  `card_number` TEXT,
  `date_valid_from` TIME,
  `date_valid_to` TIME,
  `other_card_details` TEXT
);

CREATE TABLE `Financial_Transactions` (
  `transaction_id` NUMBER,
  `previous_transaction_id` NUMBER,
  `account_id` NUMBER,
  `card_id` NUMBER,
  `transaction_type` TEXT,
  `transaction_date` TIME,
  `transaction_amount` NUMBER,
  `transaction_comment` TEXT,
  `other_transaction_details` TEXT,
  FOREIGN KEY (`account_id`) REFERENCES `Accounts`(`account_id`),
  FOREIGN KEY (`card_id`) REFERENCES `Customers_Cards`(`card_id`)
);

### Question:
What are the different customer ids, and how many cards does each one hold?

### SQL:

```
</details>

---

## Sample 15/20: `aircraft` (Complexity: **extra-hard**)
- **Token Count**: 468 tokens- **Sanity Checks**: PASSED
**Question**: find the name and age of the pilot who has won the most number of times among the pilots who are younger than 30.
**Target SQL**:
```sql
SELECT t1.name ,  t1.age FROM pilot AS t1 JOIN MATCH AS t2 ON t1.pilot_id  =  t2.winning_pilot WHERE t1.age  <  30 GROUP BY t2.winning_pilot ORDER BY count(*) DESC LIMIT 1
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `pilot` (
  `Pilot_Id` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Age` NUMBER
);

CREATE TABLE `aircraft` (
  `Aircraft_ID` NUMBER PRIMARY KEY,
  `Aircraft` TEXT,
  `Description` TEXT,
  `Max_Gross_Weight` TEXT,
  `Total_disk_area` TEXT,
  `Max_disk_Loading` TEXT
);

CREATE TABLE `match` (
  `Round` NUMBER PRIMARY KEY,
  `Location` TEXT,
  `Country` TEXT,
  `Date` TEXT,
  `Fastest_Qualifying` TEXT,
  `Winning_Pilot` TEXT,
  `Winning_Aircraft` TEXT,
  FOREIGN KEY (`Winning_Pilot`) REFERENCES `pilot`(`Pilot_Id`),
  FOREIGN KEY (`Winning_Aircraft`) REFERENCES `aircraft`(`Aircraft_ID`)
);

CREATE TABLE `airport` (
  `Airport_ID` NUMBER PRIMARY KEY,
  `Airport_Name` TEXT,
  `Total_Passengers` NUMBER,
  `%_Change_2007` TEXT,
  `International_Passengers` NUMBER,
  `Domestic_Passengers` NUMBER,
  `Transit_Passengers` NUMBER,
  `Aircraft_Movements` NUMBER,
  `Freight_Metric_Tonnes` NUMBER
);

CREATE TABLE `airport_aircraft` (
  `ID` NUMBER,
  `Airport_ID` NUMBER PRIMARY KEY,
  `Aircraft_ID` NUMBER,
  FOREIGN KEY (`Aircraft_ID`) REFERENCES `aircraft`(`Aircraft_ID`),
  FOREIGN KEY (`Airport_ID`) REFERENCES `airport`(`Airport_ID`)
);

### Question:
find the name and age of the pilot who has won the most number of times among the pilots who are younger than 30.

### SQL:

```
</details>

---

## Sample 16/20: `hr_1` (Complexity: **medium**)
- **Token Count**: 503 tokens- **Sanity Checks**: PASSED
**Question**: Give the country id and corresponding count of cities in each country.
**Target SQL**:
```sql
SELECT country_id ,  COUNT(*) FROM locations GROUP BY country_id
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `regions` (
  `REGION_ID` NUMBER PRIMARY KEY,
  `REGION_NAME` TEXT
);

CREATE TABLE `countries` (
  `COUNTRY_ID` TEXT PRIMARY KEY,
  `COUNTRY_NAME` TEXT,
  `REGION_ID` NUMBER,
  FOREIGN KEY (`REGION_ID`) REFERENCES `regions`(`REGION_ID`)
);

CREATE TABLE `departments` (
  `DEPARTMENT_ID` NUMBER PRIMARY KEY,
  `DEPARTMENT_NAME` TEXT,
  `MANAGER_ID` NUMBER,
  `LOCATION_ID` NUMBER
);

CREATE TABLE `jobs` (
  `JOB_ID` TEXT PRIMARY KEY,
  `JOB_TITLE` TEXT,
  `MIN_SALARY` NUMBER,
  `MAX_SALARY` NUMBER
);

CREATE TABLE `employees` (
  `EMPLOYEE_ID` NUMBER PRIMARY KEY,
  `FIRST_NAME` TEXT,
  `LAST_NAME` TEXT,
  `EMAIL` TEXT,
  `PHONE_NUMBER` TEXT,
  `HIRE_DATE` TIME,
  `JOB_ID` TEXT,
  `SALARY` NUMBER,
  `COMMISSION_PCT` NUMBER,
  `MANAGER_ID` NUMBER,
  `DEPARTMENT_ID` NUMBER,
  FOREIGN KEY (`JOB_ID`) REFERENCES `jobs`(`JOB_ID`),
  FOREIGN KEY (`DEPARTMENT_ID`) REFERENCES `departments`(`DEPARTMENT_ID`)
);

CREATE TABLE `job_history` (
  `EMPLOYEE_ID` NUMBER PRIMARY KEY,
  `START_DATE` TIME,
  `END_DATE` TIME,
  `JOB_ID` TEXT,
  `DEPARTMENT_ID` NUMBER,
  FOREIGN KEY (`JOB_ID`) REFERENCES `jobs`(`JOB_ID`),
  FOREIGN KEY (`DEPARTMENT_ID`) REFERENCES `departments`(`DEPARTMENT_ID`),
  FOREIGN KEY (`EMPLOYEE_ID`) REFERENCES `employees`(`EMPLOYEE_ID`)
);

CREATE TABLE `locations` (
  `LOCATION_ID` NUMBER PRIMARY KEY,
  `STREET_ADDRESS` TEXT,
  `POSTAL_CODE` TEXT,
  `CITY` TEXT,
  `STATE_PROVINCE` TEXT,
  `COUNTRY_ID` TEXT,
  FOREIGN KEY (`COUNTRY_ID`) REFERENCES `countries`(`COUNTRY_ID`)
);

### Question:
Give the country id and corresponding count of cities in each country.

### SQL:

```
</details>

---

## Sample 17/20: `musical` (Complexity: **medium**)
- **Token Count**: 179 tokens- **Sanity Checks**: PASSED
**Question**: Show different nominees and the number of musicals they have been nominated.
**Target SQL**:
```sql
SELECT Nominee ,  COUNT(*) FROM musical GROUP BY Nominee
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `musical` (
  `Musical_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Year` NUMBER,
  `Award` TEXT,
  `Category` TEXT,
  `Nominee` TEXT,
  `Result` TEXT
);

CREATE TABLE `actor` (
  `Actor_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Musical_ID` NUMBER,
  `Character` TEXT,
  `Duration` TEXT,
  `age` NUMBER,
  FOREIGN KEY (`Musical_ID`) REFERENCES `actor`(`Actor_ID`)
);

### Question:
Show different nominees and the number of musicals they have been nominated.

### SQL:

```
</details>

---

## Sample 18/20: `musical` (Complexity: **medium**)
- **Token Count**: 172 tokens- **Sanity Checks**: PASSED
**Question**: What is the duration of the oldest actor?
**Target SQL**:
```sql
SELECT Duration FROM actor ORDER BY Age DESC LIMIT 1
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `musical` (
  `Musical_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Year` NUMBER,
  `Award` TEXT,
  `Category` TEXT,
  `Nominee` TEXT,
  `Result` TEXT
);

CREATE TABLE `actor` (
  `Actor_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Musical_ID` NUMBER,
  `Character` TEXT,
  `Duration` TEXT,
  `age` NUMBER,
  FOREIGN KEY (`Musical_ID`) REFERENCES `actor`(`Actor_ID`)
);

### Question:
What is the duration of the oldest actor?

### SQL:

```
</details>

---

## Sample 19/20: `race_track` (Complexity: **medium**)
- **Token Count**: 155 tokens- **Sanity Checks**: PASSED
**Question**: Show the race class and number of races in each class.
**Target SQL**:
```sql
SELECT CLASS ,  count(*) FROM race GROUP BY CLASS
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `race` (
  `Race_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Class` TEXT,
  `Date` TEXT,
  `Track_ID` TEXT,
  FOREIGN KEY (`Track_ID`) REFERENCES `track`(`Track_ID`)
);

CREATE TABLE `track` (
  `Track_ID` NUMBER PRIMARY KEY,
  `Name` TEXT,
  `Location` TEXT,
  `Seating` NUMBER,
  `Year_Opened` NUMBER
);

### Question:
Show the race class and number of races in each class.

### SQL:

```
</details>

---

## Sample 20/20: `small_bank_1` (Complexity: **hard**)
- **Token Count**: 210 tokens- **Sanity Checks**: PASSED
**Question**: What are the checking and savings balances in accounts belonging to Brown?
**Target SQL**:
```sql
SELECT T2.balance ,  T3.balance FROM accounts AS T1 JOIN checking AS T2 ON T1.custid  =  T2.custid JOIN savings AS T3 ON T1.custid  =  T3.custid WHERE T1.name  =  'Brown'
```
<details>
<summary>Click to view serialized Database Schema & Full Prompt</summary>

```text
You are an expert SQL engineer. Given the database schema, write the exact SQLite query that answers the user question.

### Database Schema:
CREATE TABLE `ACCOUNTS` (
  `custid` NUMBER PRIMARY KEY,
  `name` TEXT
);

CREATE TABLE `SAVINGS` (
  `custid` NUMBER PRIMARY KEY,
  `balance` NUMBER,
  FOREIGN KEY (`custid`) REFERENCES `ACCOUNTS`(`custid`)
);

CREATE TABLE `CHECKING` (
  `custid` NUMBER PRIMARY KEY,
  `balance` NUMBER,
  FOREIGN KEY (`custid`) REFERENCES `ACCOUNTS`(`custid`)
);

### Question:
What are the checking and savings balances in accounts belonging to Brown?

### SQL:

```
</details>

---

