%% Test of the network transfer

t = tcpclient("LAFL",65432)

while 1
    read(t)
end

%clear t