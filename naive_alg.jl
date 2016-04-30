# naive alg. TRNG
using Distributions

function get_pmf(xin,support)
	N = length(xin)
	x_support,counts = hist(xin,support)
	pmf = counts./sum(counts);
	x_support = x_support[1:(end-1)] + x_support[2]/2;
	return pmf,counts,x_support
end

function get_cdf(xin,support)
	pmf,counts,x_support =get_pmf(xin,support);
	return cumsum(pmf), cumsum(counts),x_support
end

function entropy(y)
	N = sum(y);
	yt = [k for k in y[y.>0]];
	return  -sum( (yt/N).*log(yt/N) )/log(length(y));
end

function entropy(x, mydict)
	~,y=hist(x,mydict);
	yt = [k for k in y[y.>0]];
	N = sum(y);
	return  -sum( (yt/N).*log(yt/N) )/log(length(y));
end

function acf(xin)
	mu = mean(xin);
	N = length(xin);
	s = [(xin-mu); zeros(N+1)];

	Sxx= (abs(fft(s)).^2)/((2*N-1)*var(s));
	Rxx = fftshift(ifft(Sxx)); 
	x = [-N:1:N;];
	return real(Rxx), x
end	

function psd(xin)
	mu = mean(xin);
	N = length(xin);
	s = [(xin-mu); zeros(N+1)];

	Sxx= (abs(fft(s)).^2)/((2*N-1)*var(s));
	return real(Sxx), linspace(-0.5,0.5,length(Sxx))
end

function chi2gf_test(x)
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

function chi2gfNormal_test(x)
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

function chi2gfNormal_test(x,mu,sigma)
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

function shuffleGA(x,rt_sel,rt_mut,rt_eli,sigma_0,gmax)
	sigma_S  = 2*sigma_0;
	N = length(x);
	g=1;
	while (sigma_S > sigma_0)&(g<gmax)
		#fitness evaluation
		sigma_S = std(psd(x)[1]);
		
		#selection
		Nsel = Int(floor(rt_sel*N));
		if mod(Nsel,2)==1
			Nsel+=1;
		end 
		xind = randperm(Int(N))[1:Nsel];
		
		#crossover
		P0ind = xind[1:2:Nsel];
		P1ind = xind[2:2:Nsel];
		
		cp = rand([1:Int(Nsel/2);]);
		
		for k=1:cp
			temp=P0ind[k];
			P0ind[k]=P1ind[k];
			P1ind[k]=temp;
		end
		
		#mutation
		for k=1:Int(Nsel/2)
			if rand()<rt_mut
				xindmut = randperm(Int(Nsel/2))[1:2];
				temp = P0ind[xindmut[1]];
				P0ind[xindmut[1]] = P1ind[xindmut[2]];
				P1ind[xindmut[2]] = temp;
			end
		end
		
		#Elitism
		Neli = Int(floor(rt_eli*Nsel));
		Ch = [P0ind; P1ind][1:Neli];
		Ch = [Ch; symdiff(xind,Ch)];
		
		y = copy(x)
		y[xind] = x[Ch];
		sigma_Ch =std(psd(y)[1]);
		if sigma_Ch < sigma_S
			x = copy(y);
			sigma_S = sigma_Ch;
		end
		g+=1
		#println(g)
	end
	return x,g
end



function st_enf(x,sigma_0);

	done = false;
	#k = 1;
	while !done
		acf_vec,~ = acf(x);
		rN = length(acf_vec)
		if chi2gfNormal_test(acf_vec[1:Int((rN-1)/2)])
			psd_vec= psd(x)[1];
			if chi2gfNormal_test(psd_vec, 1, sigma_0)
				return x;
			else
				x,k = shuffleGA(x,1.0,0.01,1.0,sigma_0,1000);
				psd_vec= psd(x)[1];
				if chi2gfNormal_test(psd_vec, 1, sigma_0)
					return x;
				else
					x = rand(length(x));
				end
			end
		else 
			x = rand(length(x));
		end
	end
	
end

function run_test()
	n=270
	for k=1:30
		@time st_enf(rand(n),0.5)
	end
end

run_test()

