%% Test of the network transfer

t = tcpclient("LAFL",65432)
%data = [0x00,0x00];

while 1
    %write(t,data)
    read(t,1,"uint16")
end

%clear t