#pragma once

#include <vector>
#include <string>
#include <algorithm>

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i = 1; i < argc; ++i)
                tokens.push_back(std::string(argv[i]));
        }
        
        const std::string& get_option(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr = std::find(tokens.begin(), tokens.end(), option);
            if (itr != tokens.end() && ++itr != tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        
        bool has_option(const std::string &option) const{
            return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};
