%% Test of the network transfer

t = tcpclient("LAFL",65433)
data = [0x00,0x00];

while 1
    write(t,data)
    read(t,1,"int16")
end

%clear t