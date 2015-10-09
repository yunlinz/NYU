p = (1/2)^10;
q = 1 - p;
s = 0;
N = 4170;
for i = 5:N
  s = s + binopdf(i,N,p);
end
printf s