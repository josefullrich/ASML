%% SRCMDL
% Source Model
%
% Function:
% [state,euv,rx,ry,ts,dy,nr_exposures,de,euvol,dy_actual] = srcmdl(x,y,z,e,d,p,k,b,dt,state)
%
% Outputs:
% euv - EUV in mJ  (max is 6.85 mJ when dt = 100, max is 6.23 mJ when DY=99.5%)
% rx  - target angle Rx in deg (-23.5deg <= rx <= 23.5deg)
% ry  - target angle Ry in deg ( 11.5deg <= ry <= 23.5deg)
% ts  - target size in um   (0.1 <= ts <= 0.9)
% dy  - die yield in %/100  (>= 0.995 %/100)
% exp - number of exposures 
% de  - die error in %/100  (<= 0.01 %/100)
%
% Inputs:
% x   - l2dx in [0 1] (opt is 0.250)
% y   - l2dy in [0 1] (opt is 0.700)
% z   - l2dz in [0 1] (opt is 0.477)
% e   - pp energy in [0 1] (opt is 0.798)
% d   - mp2pp delay in [0 1] (opt is 0.150)
% p   - drive laser energy in [0 1] (opt is 0.275)
% k   - drive laser feed-forward gain in [0 1] (opt is 0.312)
% b   - beam size in [0 1] (opt is 0.657)
% dt  - dose target per burst in mJ (open-loop set dt = 100 mJ, closed-loop is [60 65] mJ with opt is 62.5 mJ)
