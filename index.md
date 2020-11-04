 Project team: Haley Xue, Meghana Jain, Vandana Ramesh, Srikesh Srinivas, Ruiqi Zhu 

## Introduction and Background 
Airbnb is an online marketplace that allows home-owners and renters (‘hosts’) to rent out their properties (‘listings’) to guests. The hosts generally set their own prices for their listings. Although Airbnb and other rental websites provide the hosts with a set of guidelines on how to set the prices for their listings, there aren’t currently any accurate methods that help the hosts price the properties based on a large number of features.  There are quite a few paid third party software available to be able to predict the pricing range, but the hosts also need to put in their expected price of the listing, and the algorithms would change the price based on the day of the week, the season, or other outside factors.  

It is pretty important to get the Airbnb pricing right, especially in big cities around the world where competition is high and small differences in price can alter the popularity of a listing. It is necessary to find the appropriate pricing method and price range on a listing.

Through our project, we plan to learn about the features that would contribute to the price of an Airbnb listing along with the features which would contribute to a higher rating for the listing. We plan to investigate the data of five cities, New York, Boston, Amsterdam, Berlin, and Hong Kong to be able to diversify our results for pricing and ratings. 

## Problem Definition
We plan to explore the pricing and rating predictions using supervised and unsupervised machine learning techniques. Through supervised learning, we would like to predict the price and rating of an Airbnb listing. Through unsupervised learning, we would like to cluster similar listings based on chosen attributes and cluster listings based on reviews. We plan to integrate the supervised and unsupervised results ------- ***

## Data Collection
We obtained our data from Inside Airbnb[1], a website that scrapes Airbnb listings, reviews, and calendar data from multiple cities around the world.  The listings dataset for each city contains listings details (room types, prices, availabilities, host ids and names, average reviews per month, etc.), and neighborhood information. It also includes geographic information which can be used to produce map visualizations. The reviews dataset contains reviews and comments about each Airbnb listing present in the listings dataset.  

### Data Exploration 
In order to get a general understanding of the listings data, we generated some exploratory graphs to visualize patterns and groupings.  
#### Number of Listings for each Neighbourhood group 
In New York City, we have the following 5 boroughs: Manhattan, Brooklyn, Queens, Staten Island, and Bronx [2]. As seen in the graph below, most Airbnb listings are found in Manhattan and Brooklyn, which is the most densely populated/landmark heavy borough and the city's most populous borough, respectively. 
<p align="center">
    <img src="price-plot.png">
</p>

<p align="center">
    <img src="dist_prices.png">
</p>
## References 
[1] Pouya Rezazadeh Kalehbasti, Liubov Nikolenko, and Hoormazd Rezaei. Airbnb Price Prediction Using Machine Learning and Sentiment Analysis. arXiv preprint arXiv:1907.12665, 2019. 

[2] Quattrone, G., Greatorex, A., Quercia, D. et al. Analyzing and predicting the spatial penetration of Airbnb in U.S. cities. EPJ Data Sci. 7, 31 (2018). https://doi.org/10.1140/epjds/s13688-018-0156-6 

[3] Yang Li, Quan Pan, Tao Yang, and Lantian Guo. Reasonable price recommendation on Airbnb using multi-scale clustering. In 2016 35th Chinese Control Conference (CCC), pages 7038–7041. IEEE, 2016. 

[4] Zhang, Shunyuan and Lee, Dokyun and Singh, Param Vir and Srinivasan, Kannan, How Much Is an Image Worth? Airbnb Property Demand Estimation Leveraging Large Scale Image Analytics (May 25, 2017). 


## Appendix 
1. [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
2. [Boroughs of NYC] (https://en.wikipedia.org/wiki/Boroughs_of_New_York_City)

[Reviews for Inside Airbnb](https://public.opendatasoft.com/explore/dataset/airbnb-reviews)

[Airbnb Reviews](https://public.opendatasoft.com/explore/dataset/airbnb-reviews/api/)

[Airbnb new user bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)

[Austin food spots](https://data.world/owentemple/austin-restaurant-health-inspection-scores/activity)

[NYC Restaurants](https://data.cityofnewyork.us/Health/Restaurant-Grades/gra9-xbjk)

