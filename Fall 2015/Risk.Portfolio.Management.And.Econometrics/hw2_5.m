data = csvread('discrim_cleaned.csv',1,0); # loads the file, already taken logs, cleaned
y = data(:,1);

x_i = data(:,2:4);
x_ii =  data(:,[3,4]);
x_iii = data(:,2:5);
x_iv = data(:,[2,5]);

# part i

result_i = ols(y, x_i);
t_i = result.tstat(2);
p_i = 1 - tcdf(t_i, size(data,1) - 1);

# part ii
covariance = cov(x_ii);
correlation = cov(1,2)/prod(sqrt(diag(covariance)));
t_logincom = result.tstat(3);
p_logincome = 1 - tcdf(t_logincome, size(data,1)-1);

t_prppov = result.tstat(4);
p_prppov = 1 - tcdf(t_prppov, size(data,1)-1);

# part iii

result_iii = ols(y, x_iii);
t_loghseval = result_iii.tstat(5);
p_loghseval = 1- tcdf(t_loghseval, size(data, 1) - 1);

# part iv

result_iv = ols(y, x_iv);
ssr_ur = result_iii.sige;
ssr_r = result_iv.sige;

q = 2;
df = (size(data,1) - 4 - 1);

F = (ssr_r - ssr_ur)/q/(ssr_ur/df);