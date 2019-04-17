function mn = mono(stereo)
mask = ~sum(stereo == 0, 2);
mn = sum(stereo, 2);
mn(mask) = mn(mask)/2;
end

