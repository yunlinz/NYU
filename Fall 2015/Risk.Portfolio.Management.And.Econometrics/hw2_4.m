data = csvread('wage2_cleaned.csv',1,0); # reads the csv file, note column 1 is already lo
# x1 = educ
# x2 = exper
# x3 = exper + tenure
y = data(:,1);
x = data(:,2:end);

result = ols(y,x);
t_theta = result.tstat(3);
p_theta = 1 - tcdf(t_theta, size(data,1)-1);