%% Test of the network transfer

t = tcpclient("LAFL",65432)
%data = [0x00,0x00];

while 1
    %write(t,data)
    %data = read(t,1,"uint16")
    raw_data = read(t, 2);
    
    if ~isempty(raw_data)
        data = typecast(raw_data, 'uint16');
        % Now 'data' contains the received uint16 value
        disp(data);
    else
        disp('Timeout or no data received.');
    end
    
end

%clear t