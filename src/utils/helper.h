//
// Created by ynk on 2/24/20.
//

#ifndef PROJECT_EDGE_HELPER_H
#define PROJECT_EDGE_HELPER_H

#include <vector>
#include <algorithm>

using namespace std;
struct Combinations
{
    typedef vector<int> combination_t;

    // initialize status
    Combinations(int N, int R) :
            completed(N < 1 || R > N),
            generated(0),
            N(N), R(R)
    {
        for (int c = 1; c <= R; ++c)
            curr.push_back(c);
    }

    // true while there are more solutions
    bool completed;

    // count how many generated
    int generated;

    // get current and compute next combination
    combination_t next()
    {
        combination_t ret = curr;

        // find what to increment
        completed = true;
        for (int i = R - 1; i >= 0; --i)
            if (curr[i] < N - R + i + 1)
            {
                int j = curr[i] + 1;
                while (i <= R-1)
                    curr[i++] = j++;
                completed = false;
                ++generated;
                break;
            }

        return ret;
    }

private:

    int N, R;
    combination_t curr;
};


#endif //PROJECT_EDGE_HELPER_H
