Данные модулей G2 такие же, как в папке field_src.
В этой папке температуры у модулей другие, со станции.
Здесь приклеены колонки CO, NO2, RH (неизвестно откуда).


Файл errG2000301.csv битый и не считывается пандой.

Данные по датчику G2000309:


Data columns (total 9 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   date    24329 non-null  object 
 1   H2Sop1  24329 non-null  float64
 2   H2Sop2  24329 non-null  float64
 3   SO2op1  24329 non-null  float64
 4   SO2op2  24329 non-null  float64
 5   RH      24329 non-null  float64
 6   CO      24329 non-null  float64
 7   NO2     24329 non-null  float64
 8   T       24329 non-null  float64
dtypes: float64(8), object(1)


						G2000309 SRC

                   date      H2Sop1      H2Sop2      SO2op1      SO2op2  \
15  2019-12-17 08:20:00  361.849000  330.591000  336.502000  341.770000   
16  2019-12-17 08:40:00  361.168000  331.048000  336.650500  341.766500   
17  2019-12-17 09:00:00  361.440500  331.593500  336.541000  341.707000   
18  2019-12-17 09:20:00  361.241000  332.163000  336.366000  341.875500   
19  2019-12-17 09:40:00  360.688000  332.620000  335.430000  341.822500  

            T  
15  -8.577500  
16  -8.510000  
17  -8.563000  
18  -8.679000  
19  -8.760500 

						G2000309 SRC METEO

                  date      H2Sop1      H2Sop2      SO2op1      SO2op2  \
15  2019-12-17 08:20:00  361.849000  330.591000  336.502000  341.770000   
16  2019-12-17 08:40:00  361.168000  331.048000  336.650500  341.766500   
17  2019-12-17 09:00:00  361.440500  331.593500  336.541000  341.707000   
18  2019-12-17 09:20:00  361.241000  332.163000  336.366000  341.875500   
19  2019-12-17 09:40:00  360.688000  332.620000  335.430000  341.822500 

        RH           CO         NO2         T  
15  59.1210   422.915740   25.443249 -10.8145  
16  59.5385   397.266307   25.527881 -10.8375  
17  60.1665   436.295844   27.084294 -10.8870  
18  60.4675   421.227041   26.908014 -10.9330  
19  60.5270   473.967660   31.601579 -10.9935  