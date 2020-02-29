classdef data_gen < handle
    properties
        R
        M
        company_list
        hist_prices
        num_segments
        hist_len
        X
    end
    methods
        function data = data_gen(R, M, companies)
            data.company_list = companies;
                        
            date_shift = 0;
            current_day = today + date_shift;
            datetime.setDefaultFormats('defaultdate','yyyy-MM-dd')
            
            company_vec = [];
            for company = transpose(companies)
                company_vec = [company_vec; hist_stock_data(current_day - 200, ...
                current_day + 1, company, 'frequency','d')];
            end
            company_vec = transpose(company_vec);
            
            hist_prices = gpuArray([]);
            company_list = [];
            for company = company_vec
                hist_prices = [hist_prices gpuArray(company.Open)];
            end
            hist_prices = transpose(hist_prices);
            data.hist_prices = hist_prices;
            
            data.segment_X(R, M);
        end
        function t_X = get_company_m(data, ticker)
            ind = find(data.company_list == ticker);
            t_X = reshape(data.X(ind(1), :, :), [data.num_segments, data.M]);
        end
        function segment_X(data, R, M)
            hist_prices = data.hist_prices;
            
            hist_len = size(hist_prices, 2);
            num_segments = floor((hist_len - M) / R);
            %while hist_len - (num_segments*R) < M
            %    num_segments = num_segments - 1;
            %end

            X = gpuArray(zeros(size(hist_prices, 1), M, num_segments));
            for i = 1:num_segments
                ind = R*(i - 1) + 1;
                X(:, :, i) = hist_prices(:, ind:ind + M - 1);
            end
            
            data.R = R;
            data.M = M;
            data.hist_len = hist_len;
            data.num_segments = num_segments;
            data.X = X;
        end
    end
end