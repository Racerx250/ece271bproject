if ~exist('data','var')
    data = data_gen(5, 5, ["AAPL"; "SPY"; "QQQ"]);
end
aapl_m = data.get_company_m("AAPL");
data.segment_X(3, 20);