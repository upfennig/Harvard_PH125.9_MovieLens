# Import libraries
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(knitr)

# Download movieLens data from source
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Convert movie file into dataframe
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Join movies data frame with ratings and save into a new data frame
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# edx = train set 
# validation = test set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Data exploration of train data set 
dim(edx)
kable(head(edx))
kable(summary(edx))
edx %>% distinct(movieId) %>% summarize(movie = n())
edx %>% distinct(userId) %>% summarize(user = n())

# movie rating: basic descriptive summary
kable(edx %>% summarize(avg = mean(rating), median = median(rating), sd = sd(rating)))

# top 10 most often rated movies and their corresponding rating
kable(edx %>% select(movieId, title, rating) %>% group_by(title) %>% summarize(n = n(), avg = mean(rating)) %>% arrange(desc(n)) %>% top_n(10, n))

# movie rating density
rating_per_movie <- edx %>% select(movieId, title, rating) %>% group_by(title) %>% summarize(n = n(), avg = mean(rating))
kable(summary(rating_per_movie))
ggplot(rating_per_movie, aes(n)) + 
  geom_density(fill = "red", alpha = 0.25) +
  scale_x_log10() +
  labs(x = "Number of ratings (movie level)", y = "density")

rating_per_movie %>% mutate(group_id = if_else(n >843, 1,2)) %>% group_by(group_id) %>% summarize(movie = n())

rating_per_movie_grouped <- rating_per_movie %>% mutate(group_id = if_else(n >843, 1,2))
rating_per_movie_grouped %>% group_by(group_id) %>% summarize(avg = mean(avg))

# scatterplot movie rating
edx %>% group_by(movieId) %>% summarize(n = n(), title = title[1], rating = mean(rating)) %>%
  ggplot(aes(n, rating)) +
  geom_point(size=0.75, alpha = 1/5) +
  geom_smooth()

# rating_per_user / user rating density
rating_per_user <- edx %>% select(userId, rating) %>% group_by(userId) %>% summarize(n = n(), avg = mean(rating))
kable(summary(rating_per_user))
ggplot(rating_per_user, aes(n)) + 
  geom_density(fill = "blue", alpha = 0.25) +
  scale_x_log10() +
  labs(x = "Number of ratings (user level)", y = "density")

rating_per_user %>% mutate(group_id = if_else(n >128, 1,2)) %>% group_by(group_id) %>% summarize(user = n())

# scatterplot user rating
edx %>% group_by(userId) %>% summarize(n = n(), user = userId[1], rating = mean(rating)) %>%
  ggplot(aes(n, rating)) +
  geom_point(size=0.75, alpha = 1/5) +
  geom_smooth()

# due to high data volume, select sample
n <- sample(unique(edx$userId), 200)
edx_sample <- edx %>% filter(userId %in% n)


# RMSE_function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# average_rating_model
mu_hat <- mean(edx$rating)
mu_hat

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

# RMSE results table 
rmse_results <- tibble(method = "Average rating model", RMSE = naive_rmse)

# LSE movie effect
mu <- mean(edx$rating)
movie_avgs <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

# plot_LSE
movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(bins = 10, color = I("black"))

# movie_effect_model
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>% pull(b_i)
RMSE(predicted_ratings, validation$rating)

# Add result of movie effect model to table, model_1_rmse
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie effect model", RMSE=model_1_rmse))
rmse_results %>% knitr::kable()


# LSE user + movie effect
user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# movie_user_effect_model
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, validation$rating)

# model_2_rmse
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + user effect model", RMSE=model_2_rmse))
rmse_results %>% knitr::kable()

# regularization_lambda
lambda <- seq(0, 10, 0.25)

rmses <- sapply(lambda, function(l){
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by='movieId') %>% 
    left_join(b_u, by = 'userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

# plot_lambda
qplot(lambda, rmses)  

# optimal_lambda 
lambda[which.min(rmses)]

# RMSE_regularization
min(rmses)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized movie + user effect model",  
                                 RMSE = min(rmses)))

# results, overview of RMSE across the different model optimization
rmse_results %>% knitr::kable()