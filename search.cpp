
    const int INPUT_SIZE = 5;
    const int SEARCH_RADIUS = 1;
    const int DEG_LIMIT = 3;

    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(-1, 1);

    std::vector<int64_t> test_poly(1ULL << INPUT_SIZE);
    for (size_t i = 0; i < test_poly.size(); ++i) {
        if (util::popcount(i) <= DEG_LIMIT) {
            test_poly[i] = -SEARCH_RADIUS;
        }
    }

    size_t cnt = 0;
    while (true) {
        // f.balanced_randomize();
        size_t i = 0;
        for (; i < test_poly.size(); ++i) {
            if (util::popcount(i) <= DEG_LIMIT) {
                if (test_poly[i] < SEARCH_RADIUS) {
                    ++test_poly[i];
                    break;
                } else {
                    test_poly[i] = -SEARCH_RADIUS;
                }
            }
        }
        if (i >= test_poly.size()) {
            std::cerr << "BAD, NOTHING FOUND\n";
            return 0;
        }
        BoolFun tmp = BoolFun::from_int_poly(test_poly);
        if (!tmp.invalid) {
            f = tmp;
            break;
        }
        ++cnt;
        if (cnt % 1000000 == 0)
            std::cerr << cnt << " ";
    }
