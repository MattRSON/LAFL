%% Test of the network transfer

t = tcpclient("LAFL",65432)
data = uint8(1:10);
i=0;
while i < 10
    write(t,data);
    read(t)
    i = i+1;
end

%clear t