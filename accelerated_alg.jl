# accelerated alg. TRNG
using Distributions

@everywhere function get_pmf(xin,support)
	N = length(xin)
	x_support,counts = hist(xin,support)
	pmf = counts./sum(counts);
	x_support = x_support[1:(end-1)] + x_support[2]/2;
	return pmf,counts,x_support
end

@everywhere function get_cdf(xin,support)
	pmf,counts,x_support =get_pmf(xin,support);
	return cumsum(pmf), cumsum(counts),x_support
end

@everywhere function entropy(y)
	N = sum(y);
	yt = [k for k in y[y.>0]];
	return  -sum( (yt/N).*log(yt/N) )/log(length(y));
end

@everywhere function entropy(x, mydict)
	~,y=hist(x,mydict);
	yt = [k for k in y[y.>0]];
	N = sum(y);
	return  -sum( (yt/N).*log(yt/N) )/log(length(y));
end

@everywhere function acf(xin)
	mu = mean(xin);
	N = length(xin);
	s = [(xin-mu); zeros(N+1)];

	Sxx= (abs(fft(s)).^2)/((2*N-1)*var(s));
	Rxx = fftshift(ifft(Sxx)); 
	x = [-N:1:N;];
	return real(Rxx), x
end	

@everywhere function psd(xin)
	mu = mean(xin);
	N = length(xin);
	s = [(xin-mu); zeros(N+1)];

	Sxx= (abs(fft(s)).^2)/((2*N-1)*var(s));
	return real(Sxx), linspace(-0.5,0.5,length(Sxx))
end

@everywhere function chi2gf_test(x)
	alpha = 0.05
	k = 5-3;
	support = [0:1/5:1;];
	
	E = (1/5)*ones(5);
	O,~,~ = get_pmf(x,support);
	
	chi2_calc = sum( ((O.-E).^2)./E );
	chi2_ref = cquantile(Chisq(k),alpha);
	
	if chi2_ref < chi2_calc
		ans = false;
	else
		ans = true;
	end
	
	return ans
end

@everywhere function chi2gfNormal_test(x)
	alpha = 0.05
	k = 5-3;
	support = [-1:2/7:1;];
	dis = fit(Normal, x)
	
	E = pdf(Normal(dis.μ, dis.σ),support[1:(end-1)] + (support[2]-support[1])/2 )
	O,~,~ = get_pmf(x,support);
	
	chi2_calc = sum( ((O.-E).^2)./E );
	chi2_ref = cquantile(Chisq(k),alpha);
	
	if chi2_ref < chi2_calc
		ans = false;
	else
		ans = true;
	end
	
	return ans
end

@everywhere function chi2gfNormal_test(x,mu,sigma)
	alpha = 0.05
	k = 5-3;
	support = [-1:2/7:1;];
	
	E = pdf(Normal(mu, sigma),support[1:(end-1)] + (support[2]-support[1])/2 )
	O,~,~ = get_pmf(x,support);
	
	chi2_calc = sum( ((O.-E).^2)./E );
	chi2_ref = cquantile(Chisq(k),alpha);
	
	if chi2_ref < chi2_calc
		ans = false;
	else
		ans = true;
	end
	
	return ans
end

@everywhere function selectionFisherYates(x)
	N =length(x)
	for i in [1:(N-1);]
		j = rand([1:(N-i);]);
		x[i], x[i+j] = x[i+j], x[i];
	end
	return x
end 

@everywhere function crossover(N)
	p1_ind = shuffle([1:N;])
	p2_ind = shuffle([1:N;])
	childVec_ind = zeros(Int64,N)
	ini_Pt = rand( [1:(N - 1 );]);
	end_Pt = rand( [(ini_Pt + 1):N;]);

	A1 = zeros(Int64, end_Pt-ini_Pt + 1 )
	A1 = p1_ind[ini_Pt:end_Pt]
	A0 = setdiff(p2_ind, A1 )[1:(ini_Pt-1)]
	A2 = setdiff(p2_ind, [A1;A0] )
	childVec_ind = [A0; A1; A2]
	return childVec_ind
end

@everywhere function mutation(child1, prob_mutation, N)
	for k =1: length(child1)
		if rand() <= prob_mutation
			i = rand([1:(N-1);] );
			j = rand([1:(N-i);] );
			child1[k][i], child1[k][i+j] = child1[k][i+j],child1[k][i];
		end
	end
	return child1
end

@everywhere function replacement(P,x,Ch,Ext,xE,N)
	
	P_val = [std(psd(x[ P[k] ])[1]) for k=1:N]; 
	Ch_val = [std(psd(x[ Ch[k] ])[1]) for k=1:N]; 
	Ext_val = [std(psd(xE[ Ext[k] ])[1]) for k=1:N]; 
	
	sigma_P,P_ind = findmin(P_val);
	sigma_Ch,Ch_ind = findmin(Ch_val);
	sigma_Ext,Ext_ind = findmin(Ext_val);	
	
	if sigma_Ext < minimum([sigma_P,sigma_Ch])
		x = copy(xE);
		P = copy(Ext);
		P_ind =Ext_ind;
		sigma_P = sigma_Ext;
	else
		ind = sortperm( [P_val;Ch_val])[1:N];
		P = ([P;Ch])[ind];
		P_ind = 1;
		sigma_P = minimum([P_val;Ch_val]);
	end
	
	return x,P,P_ind,sigma_P
end
		
@everywhere function shuffleGApar(x,sigma_0, N)

	#selection
	P = [ selectionFisherYates( [1:length(x);] ) for k=1:N];
	#fitness evaluation
	P_val = [std(psd(x[ P[k] ])[1]) for k=1:N];
	sigma_best, P_ind = findmin(P_val);
	x_best = copy(x[P[P_ind]]);
	
	k=1;
	while (sigma_best > sigma_0) & (k <=100)
		Ch = [crossover(length(x)) for k=1:N];											# crossover
		Ch = mutation(Ch, 0.05, Int(length(x)/2));																		# mutation
		x_Ext = rand(N);																						# selection of new x_Ext
		Ext = [ selectionFisherYates( [1:N;] ) for k=1:N];
		x, P,P_ind, sigma_best = replacement(P,x,Ch,Ext,x_Ext,N);		# replacement
		k +=1;		
 end
 #println("workers=$(workers()) .... myid=$(myid())")
	#Sh[myid()] = 1;
 return x[P[P_ind]]
#		Sh[1]=1;
# 	if (sigma_best <= sigma_0)
# 		return x[P[P_ind]]
# 	else
# 		return 0
# 	end
end



@everywhere function st_enf(x,sigma_0)
	N = length(x);
	np = nworkers();
	xpar = [x for i=1:np];
	sigma_0par = [sigma_0 for i=1:np];
	Npar = [N for i=1:np];
	
	return pmap((a,b,c) -> shuffleGApar(a,b,c), xpar, sigma_0par, Npar)
end

@everywhere function run_test()
	n=6
	for k=1:1024
		@time st_enf(rand(2^n), 0.5)
	end
end



Sh = SharedArray(Int, (nworkers()+2,1), init=Sh->Sh[localindexes(Sh)] = 0);
#run_test()

# function run_test()
# 	n=8
# 	for k=1:1024
# 		@time st_enf(rand(2^n),0.5)
# 	end
# end
#
# run_test()

