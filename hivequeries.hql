
-- HIVE query to filter for only data of Disney from the NYSE stock prices
create table final_Project AS 
SELECT to_date(date) AS date_,
         open,
         close
FROM price_ad
WHERE symbol = "DIS" 


-- HIVE query to get additional columns to make visualization easier in Tableau
create table final_project.OHE ASSELECT *,
        
    CASE
    WHEN D_diff >= 0 THEN
    "UP"
    ELSE "DOWN"
    END AS Growth, concat(year(date_),'-',weekofyear(date_)) AS y_month, year(date_) AS year_num, month(date_) AS month_num, weekofyear(date_) AS week_num
FROM 
  (SELECT to_date(date) AS date_,
         symbol,
         open,
         high,
         low,
         close,
         volume,
         (close-open) AS D_diff,
         (high-low) AS D_range
  FROM final_project.price_ad
  WHERE symbol="DIS" )a