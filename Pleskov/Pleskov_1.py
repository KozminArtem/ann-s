### https://medium.com/@ppleskov/машинное-обучение-это-весело-часть-1-b52f24b40c28
##  Pavel Pleskov in medium
#   lection 1 

"""
	# Основные правила для оценки стоимости дома
	# Пример первый
def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):
	price = 0
	# In my area, the average house costs $200 per sqft
	price_per_sqft = 200
	if neighborhood == "hipsterton": # but some areas cost a bit more
		price_per_sqft = 400
	elif neighborhood == "skid row": # and some areas cost less
		price_per_sqft = 100

	# start with a base price estimate based on how big the place is
	price = price_per_sqft * sqft
	
	# now adjust our estimate based on the number of bedrooms
	if num_of_bedrooms == 0:
		# Studio apartments are cheap
		price = price - 20000
	else:
		# places with more bedrooms are usually
		# more valuable
		price = price + (num_of_bedrooms * 1000)
	return price

print(estimate_house_sales_price(3,2000,"Normaltown"))
"""
	# Пример второй

def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):
	price = 0.
	# a little pinch of this
	price += num_of_bedrooms * .841231951398213
	# and a big pinch of that
	price += sqft * 1231.1231231
	# maybe a handful of this
	price += neighborhood * 2.3242341421
	# and finally, just a little extra salt for good measure
	price += 201.23432095
	return price

print(estimate_house_sales_price(3,2000,100))