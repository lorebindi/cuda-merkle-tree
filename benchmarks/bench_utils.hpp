#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <ctime>
#include <sys/stat.h>

using namespace std;

// Return a timestamp like "20250419_153042"
inline string timestamp_str() {
    time_t now = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&now));
    return string(buf);
}

// Writes header + row on CSV file and on stdout 
struct BenchmarkTable {
    vector<string> headers;
    vector<vector<string>> rows;
    string title;

    BenchmarkTable(const string& title, vector<string> headers)
        : title(title), headers(move(headers)) {}

    void add_row(vector<string> row) {
        rows.push_back(move(row));
    }

    void print_stdout() const {
        int col_w = 20;
        cout << "\n=== " << title << " ===\n";
        cout << left;
        for (auto& h : headers)
            cout << setw(col_w) << h;
        cout << "\n" << string(col_w * headers.size(), '-') << "\n";
        for (auto& row : rows) {
            for (auto& cell : row)
                cout << setw(col_w) << cell;
            cout << "\n";
        }
        cout << string(col_w * headers.size(), '-') << "\n";
    }

    void write_csv(const string& filepath) const {
         
        auto slash = filepath.rfind('/');
        if (slash != string::npos) {
            string dir = filepath.substr(0, slash);
            mkdir(dir.c_str(), 0755); // no-op se esiste già
        }

        ofstream f(filepath);
        if (!f.is_open()) {
            cerr << "[warn] impossibile aprire " << filepath << "\n";
            return;
        }
        for (size_t i = 0; i < headers.size(); i++) {
            f << headers[i];
            if (i + 1 < headers.size()) f << ",";
        }
        f << "\n";
        for (auto& row : rows) {
            for (size_t i = 0; i < row.size(); i++) {
                f << row[i];
                if (i + 1 < row.size()) f << ",";
            }
            f << "\n";
        }
        cout << "[csv] scritto: " << filepath << "\n";
    }

    void dump(const string& out_dir = ".") const {
        print_stdout();
        string safe_title = title;
        for (char& c : safe_title)
            if (c == ' ' || c == '/' || c == '\\') c = '_';
        write_csv(out_dir + "/" + safe_title + "_" + timestamp_str() + ".csv");
    }
};

/* Holds mean, standard deviation and coefficient of variation for a benchmark run */
struct BenchResult {
    uint64_t mean;
    double   stddev;
    double   cv;      // stddev / mean * 100  (%)

    /* Builds a BenchResult from a vector of raw samples */
    static BenchResult from_samples(const vector<uint64_t>& samples) {
        BenchResult r{};
        for (auto s : samples) r.mean += s;
        r.mean /= samples.size();

        double variance = 0.0;
        for (auto s : samples) {
            double diff = (double)s - (double)r.mean;
            variance += diff * diff;
        }
        r.stddev = sqrt(variance / samples.size());
        r.cv     = (r.mean > 0) ? (r.stddev / r.mean * 100.0) : 0.0;
        return r;
    }
};