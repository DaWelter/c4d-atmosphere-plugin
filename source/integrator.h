#pragma once

#include <cmath>
#include <cassert>
#include <algorithm>

namespace numerical_integrator
{

template<class T>
inline double norm(const T& a)
{
    return std::abs(a);
}

template<class T>
inline T zero()
{
    return T{};
}

template<class T>
inline bool all_abs_less_eq(const T& a, double b_mult, const T& b)
{
	return std::abs(a) <= b_mult*abs(b);
}


namespace detail
{

using numerical_integrator::zero;
using numerical_integrator::norm;

template<class T>
struct SimpsonQuadResult
{
    T i;
    T fm;
    double m;
};

template<class T, class Func>
inline SimpsonQuadResult<T> simpson_quadrature(double a, double b, const T& fa, const T& fb, const Func &f)
{
    const double m = 0.5*(a+b);
    const auto fm = f(m);
    const auto i = (1./6.) * std::abs(b - a) * (fa + 4. * fm + fb);
    return { i, fm, m };
}

template<class Func>
inline double simpson_recursive(double a, double m, double b, double fa, double fm, double fb, double eps, double whole, const Func &f)
{
    const auto left = simpson_quadrature<double>(a, m, fa, fm, f);
    const auto right = simpson_quadrature<double>(m, b, fm, fb, f);
    const double delta = left.i + right.i - whole;
    if (std::abs(delta) <= 15 * eps)
        return left.i + right.i + delta / 15.;
    return  simpson_recursive(a, left.m, m, fa, left.fm, fm, eps*0.5, left.i, f) +
            simpson_recursive(m, right.m, b, fm, right.fm, fb, eps*0.5, right.i, f);
}

template<class Func>
inline double simpson_integration(double a, double b, double eps, Func &&f)
{
    const double fa = f(a);
    const double fb = f(b);
    const auto whole = detail::simpson_quadrature<double>(a, b, fa, fb, f);
    return detail::simpson_recursive(a, whole.m, b, fa, whole.fm, fb, eps, whole.i, f);
}

template<class T>
struct RecResult
{
    T integral;
    int num_points;
};

template<class T>
struct Buffer
{
    double* xs;
    T* is;
    int size;
};

template<class T>
Buffer<T> advanced(Buffer<T>& b, int num)
{
    assert (num <= b.size);
    return { b.xs+num, b.is+num, b.size-num };
}


template<class T, class Func>
inline RecResult<T> simpson_recursive(double a, double m, double b, const T& fa, const T& fm, const T& fb, double eps, int max_levels, const T& whole, Buffer<T> buffer, const Func &f)
{
    const auto left = simpson_quadrature<T>(a, m, fa, fm, f);
    const auto right = simpson_quadrature<T>(m, b, fm, fb, f);
    const T delta = left.i + right.i - whole;
	const T accurate_result = left.i + right.i + (1. / 15.)*delta;
    if (all_abs_less_eq(delta, eps, accurate_result) || buffer.size <= 1 || max_levels <= 1)
    {
        
        buffer.xs[0] = b;
        buffer.is[0] = buffer.is[-1] + accurate_result;
        return { accurate_result, 1 };
    }
    else
    {
        const auto left_rec = simpson_recursive<T>(a, left.m, m, fa, left.fm, fm, eps, max_levels-1, left.i, buffer, f);
        const auto right_rec = simpson_recursive<T>(m, right.m, b, fm, right.fm, fb, eps, max_levels-1, right.i, advanced<T>(buffer, left_rec.num_points), f);
        return { left_rec.integral+right_rec.integral, left_rec.num_points+right_rec.num_points };
    }
}


template<class T, class Func>
inline RecResult<T> simpson_integration(double a, double b, double stepsize, double eps, int max_levels, Buffer<T> buffer, Func &&f)
{
	assert(buffer.size >= 2);
    const int num_segments = std::min(std::max(1, int((b-a) / stepsize + 0.5)), buffer.size-1);
    stepsize = (b-a) / num_segments;
    //eps /= num_segments;
    RecResult<T> total{ zero<T>(), 1 };
    buffer.xs[0] = a;
    buffer.is[0] = zero<T>();
    buffer = advanced(buffer, 1);
    T fsa = f(a);
    double sa = a;
    for (int i = 0; i<num_segments; ++i)
    {
        const double sb = sa + stepsize;
        const T fsb = f(sb);
        // Assume buffer_size == num_segments+1. Then free_elems should become constant 1 because each step can only add one element.
        // For i==0: num_segments+1 - num_segments - 0 == 1  ok.
        const auto this_segment_buffer = Buffer<T>{ buffer.xs, buffer.is, buffer.size - (num_segments - i) };
        const auto whole = detail::simpson_quadrature<T>(sa, sb, fsa, fsb, f);
        const auto segment = detail::simpson_recursive<T>(sa, whole.m, sb, fsa, whole.fm, fsb, eps, max_levels, whole.i, this_segment_buffer, f);
        buffer = advanced(buffer, segment.num_points);
        total.integral += segment.integral;
        total.num_points += segment.num_points;
        sa = sb;
        fsa = fsb;
    }
    return total;
}

namespace fejer_quadrature_points
{

static constexpr int num_points = 7;
static constexpr double x_low_order[3] = { 0.146, 0.5,   0.854}; // Superflous, but have it anyway.
static constexpr double w_low_order[3] = { 0.286, 0.429, 0.286};
static constexpr double x_high_order[num_points] = { 0.038, 0.146, 0.309, 0.5, 0.691, 0.854, 0.962 };
static constexpr double w_high_order[num_points] = { 0.074, 0.142, 0.184, 0.2, 0.184, 0.142, 0.074 };

}

template<class T>
struct ValErr
{
    T value;
    T err;
};


template<class T, class Func>
inline ValErr<T> fejer_quadrature(double a, double b, Func &&f, double shift=0.)
{
    using namespace fejer_quadrature_points;
    const double l = b-a;
    ValErr<T> out { zero<T>(), zero<T>() };
    for (int i=0; i<num_points; ++i)
    {
		const double loc = x_high_order[i] + shift * 0.5*w_high_order[i];
        const T fval = f(a+loc*l);
        out.value += w_high_order[i]*fval;
        out.err += (i&1) ? (w_low_order[i/2]*fval) : zero<T>();
    }
    out.value *= l;
    out.err = l*out.err - out.value;
    return out;
}


template<class T, class Func>
inline T fejer_recursive(double a, double b, double eps, int max_levels, Func &&f)
{
    const auto integral = fejer_quadrature<T>(a, b, f);
    if (all_abs_less_eq(integral.err, eps, integral.value) || max_levels <= 1)
    {
		return integral.value;
    }
    else
    {
		return  fejer_recursive<T>(a, 0.5*(a + b), eps, max_levels - 1, f) +
			fejer_recursive<T>(0.5*(a + b), b, eps, max_levels - 1, f);
    }
}


template<class T, class Func>
inline T fejer(double a, double b, double stepsize, double eps, int max_levels, Func &&f)
{
    const int num_segments = std::max(1, int((b-a) / stepsize + 0.5));
    stepsize = (b-a) / num_segments;
    T result = zero<T>();
    double sa = a;
    for (int i=0; i<num_segments; ++i)
    {
        result += fejer_recursive<T>(sa, sa+stepsize, eps, max_levels, f);
        sa += stepsize;
    }
    return result;
}

/*-------------------------------------------
    Fejer rule with global error control
---------------------------------------------*/

template<class T>
struct IntegralSegment
{
	ValErr<T> integral;
	double a, b;
	double abs_err;
	int level;
};

struct SegmentCompare
{
	template<class T>
	bool operator()(const IntegralSegment<T> &a, const IntegralSegment<T> &b) const
	{
		return a.abs_err < b.abs_err;
	}
};

template<class T, int N>
class StaticBuffer
{
	T buffer[N];
	int count = 0;

public:
	T* data()
	{
		return buffer;
	}

	const T* data() const
	{
		return buffer;
	}

	void push_back(const T& elem)
	{
		assert(count < N);
		buffer[count++] = elem;
	}

	void pop_back()
	{
		--count;
		assert(count >= 0);
	}

	T& operator[](int i) {
		assert(0 <= i && i < count);
		return buffer[i];
	}

	const T& operator[](int i) const {
		assert(0 <= i && i < count);
		return buffer[i];
	}

	const T& back() const {
		return data()[count - 1];
	}

	int size() const {
		return count;
	}

	T* begin() { return data(); }
	T* end() { return  data() + count; }
};


template<class T, int STATIC_BUFFER_SIZE, class Func>
inline T fejer_global(double a, double b, double stepsize, double eps, int max_levels, Func &&f, double shift = 0.)
{
	// Warning: I limit the number of segments to the size of the buffer.
	const int num_base_segments = std::min(std::max(1, int((b - a) / stepsize + 0.5)), STATIC_BUFFER_SIZE);
	stepsize = (b - a) / num_base_segments;
	
	auto make_segment = [&](double sa, double sb, int level) -> IntegralSegment<T>
	{
		auto elem = IntegralSegment<T>{};
		elem.integral = fejer_quadrature<T>(sa, sb, f, shift);
		elem.a = sa;
		elem.b = sb;
		elem.abs_err = norm(elem.integral.err);
		elem.level = level;
		return elem;
	};

	StaticBuffer<IntegralSegment<T>, STATIC_BUFFER_SIZE> segments;

	T result = zero<T>();
	T total_err = zero<T>();
	double sa = a;
	for (int i = 0; i < num_base_segments; ++i)
	{
		const double sb = sa + stepsize;
		const auto elem = make_segment(sa, sb, 1);
		total_err += elem.integral.err;
		result += elem.integral.value;
		segments.push_back(elem);
		sa = sb;
	}

	if (all_abs_less_eq(total_err, eps, result))
		return result;

	std::make_heap(segments.begin(), segments.end(), SegmentCompare{});

	while (segments.size() && segments.size() < STATIC_BUFFER_SIZE)
	{
		// Get segment with largest error
		std::pop_heap(segments.begin(), segments.end(), SegmentCompare{});
		auto seg = segments.back();
		segments.pop_back();

		if (seg.level >= max_levels)
			continue;

		double m = 0.5*(seg.a + seg.b);
		auto seg_l = make_segment(seg.a, m, seg.level+1);
		auto seg_r = make_segment(m, seg.b, seg.level+1);

		result += seg_l.integral.value + seg_r.integral.value - seg.integral.value;
		total_err += seg_l.integral.err + seg_r.integral.err - seg.integral.err;

		if (all_abs_less_eq(total_err, eps, result))
			break;

		segments.push_back(seg_l);
		std::push_heap(segments.begin(), segments.end(), SegmentCompare{});
		segments.push_back(seg_r);
		std::push_heap(segments.begin(), segments.end(), SegmentCompare{});
	}

	return result;
}



} // detail

using detail::simpson_integration;
using detail::fejer;
using detail::fejer_global;
using detail::Buffer;
using detail::IntegralSegment;

}


namespace numerics
{

template<class Func>
inline double bisect_root(double a, double b, Func &&func, double eps_x, double eps_y)
{
	double fb = func(b);
	if (fb < 0.)
		return std::numeric_limits<double>::infinity();
	double fa = func(a);
	if (fa > 0.)
		return -std::numeric_limits<double>::infinity();
	while ((b - a > eps_x) || (fb - fa) > eps_y)
	{
		assert(fa <= 0. && fb >= 0.);
		double center = 0.5*(b + a);
		double fc = func(center);
		///                         fb    
		///             fc?            
		///  a --------- c --------- b 
		///             fc?
		/// fa
		if (fc > 0.)
		{
			b = center;
			fb = fc;
		}
		else
		{
			a = center;
			fa = fc;
		}
	}
	return 0.5*(a + b);
}

}