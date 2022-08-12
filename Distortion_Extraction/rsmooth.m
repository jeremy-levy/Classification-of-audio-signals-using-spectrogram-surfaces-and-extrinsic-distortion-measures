function z = rsmooth(y)
[n1,n2] = size(y); n = n1*n2;
N = sum([n1,n2]~=1);
Lambda = bsxfun(@plus,repmat(-2+2*cos((0:n2-1)*pi/n2),n1,1),...
-2+2*cos((0:n1-1).'*pi/n1));
W = ones(n1,n2);
zz = y;

for k = 1:6
    tol = Inf;
    while tol>1e-5
        DCTy = dct2(W.*(y-zz)+zz);
        p = fminbnd(@GCVscore,-15,38);
        s = 10^p;
        tol = norm(zz(:)-z(:))/norm(z(:));
        zz = z;
    end
    tmp = sqrt(1+16*s);
    h = (sqrt(1+tmp)/sqrt(2)/tmp)^N;
    W = bisquare(y-z,h);
end

function GCVs = GCVscore(p)
s = 10^p;
Gamma = 1./(1+s*Lambda.^2);

z = idct2(Gamma.*DCTy);
RSS = norm(sqrt(W(:)).*(y(:)-z(:)))^2;
TrH = sum(Gamma(:));
GCVs = RSS/n/(1-TrH/n)^2;

end
end

function W = bisquare(r,h)
MAD = median(abs(r(:)-median(r(:))));
u = abs(r/(1.4826*MAD)/sqrt(1-h));
W = (1-(u/4.685).^2).^2.*((u/4.685)<1);
end
