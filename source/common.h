#pragma once

#include "c4d.h"
#include "maxon/general_math.h"
#include "maxon/lib_math.h"

#include <cassert>
#include <array>

#if (API_VERSION < 23000)
#define SqrMat3 SqMat3
#endif

namespace maxon
{

	template<class T>
	constexpr T SignT(T x)
	{
		return x < (T)0 ? T{ -1 } : T{ 1 };
	}

	inline Vector64 Div(const Vector64 &a, const Vector64 &b)
	{
		return Vector64(
			a.x / b.x,
			a.y / b.y,
			a.z / b.z
		);
	}

	inline Vector64 VAbs(const Vector64 &a)
	{
		return Vector64(Abs(a.x), Abs(a.y), Abs(a.z));
	}

	inline Vector64 Exp(const Vector64 &a)
	{
		return Vector64(Exp(a.x), Exp(a.y), Exp(a.z));
	}

	inline Bool NearEqual(const Vector64 &a, const Vector64 &b, Float64 eps_abs)
	{
		const Vector64 tmp = Abs(a - b);
		return tmp.x < eps_abs && tmp.y < eps_abs && tmp.z < eps_abs;
	}

	inline Bool IsFinite(const Float64 x)
	{
		return isfinite(x);
	}

	inline Bool AllFinite(const Vector64 &v)
	{
		return IsFinite(v.x) && IsFinite(v.y) && IsFinite(v.z);
	}

	inline Float64 CopySign(Float64 a, Float64 b)
	{
		return copysign(a, b);
	}
}

using maxon::IsFinite;


template<int alignment_axis, class T>
inline maxon::SqrMat3<maxon::Vec3<T>> OrthogonalSystem(const maxon::Vec3<T> &V)
{
	using Scalar = T;
	maxon::SqrMat3<maxon::Vec3<T>> m;
	auto &Z = m[alignment_axis];
	auto &X = m[(alignment_axis + 1) % 3];
	auto &Y = m[(alignment_axis + 2) % 3];
	Z = V.GetNormalized();
	// Listing 3 in Duff et al. (2017) "Building an Orthonormal Basis, Revisited".
	Scalar sign = CopySign(Scalar(1.0), Z[2]);
	const Scalar a = Scalar(-1.0) / (sign + Z[2]);
	const Scalar b = Z[0] * Z[1] * a;
	X = maxon::Vec3<T>(Scalar(1.0f) + sign * Z[0] * Z[0] * a, sign * b, -sign * Z[0]);
	Y = maxon::Vec3<T>(b, sign + Z[1] * Z[1] * a, -Z[1]);
	return m;
}


namespace shit
{
	// Because C++11 is fairly limited ...

	template<class T>
	constexpr T Max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}

	// This is BUGGED! Gives wrong result!!
	//template<class T>
	//constexpr T MaxElement(const T* first, const T* last)
	//{
	//	return (first + 1 == last) ? *first : shit::Max(*first, MaxElement(first + 1, last));
	//}

	template<UInt64 n, UInt64 x, UInt64... numbers>
	struct MaxElementImpl
	{
		static constexpr auto Value = Max(x, MaxElementImpl<n - 1, numbers...>::Value);
	};

	template<UInt64 x>
	struct MaxElementImpl<1,x>
	{
		static constexpr auto Value = x;
	};

	template<UInt64... numbers>
	constexpr UInt64 MaxElement()
	{
		return MaxElementImpl<sizeof...(numbers), numbers...>::Value;
	}

	
}



namespace atmosphere
{

	static constexpr double LargeNumber = 1.e20;

	struct Segment
	{
		Vector64 p, v;
		Float64 dist = 0.;

		Segment(const Vector64 &p, const Vector64 &v, Float64 dist)
			: p{ p }, v{ v }, dist{ dist } {}
		Segment() = default;

		operator bool() const { return dist > 0.; }

		Vector64 PointAt(Float64 t) const { return p + t * v; }
	};


	using Segments = std::array<Segment, 2>;


	struct MediaCoefficients
	{
		Vector64 sigma_s;
		Vector64 sigma_t;
	};


	template<class VectorType = Vector64>
	static Bool RaySphereIntersection(Float64 rad, const VectorType &o, const VectorType &d, Float64 &s0, Float64 &s1)
	{
#if 1
		// https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_7.pdf
		const Float64 a = Dot(d, d);
		const Float64 bprime = -Dot(o, d);
		const VectorType tmp = o + (bprime / a) * d;
		const Float64 delta = rad * rad - tmp.GetSquaredLength();
		if (delta < 0.)
			return FALSE;
		const Float64 c = o.GetSquaredLength() - rad * rad;
		const Float64 q = bprime + static_cast<Float64>(Sign(bprime))*maxon::Sqrt(a*delta);
		s0 = c / q;
		s1 = q / a;
		if (s0 > s1)
			maxon::Swap(s0, s1);
		return TRUE;
#else
		auto b = Dot(o, d);
		auto c = Dot(o, o) - rad*rad;
		auto h = b * b - c;
		if (h < 0.0) return FALSE;
		s0 = -b - maxon::Sqrt(h);
		s1 = -b + maxon::Sqrt(h);
		return TRUE;
#endif
	}

	template<class VectorType = Vector64>
	static Bool RaySphereClip(Float64 rad, const VectorType &z, const VectorType &v, Float64 &s0, Float64 &s1)
	{
		Float64 q0, q1;
		if (!RaySphereIntersection(rad, z, v, q0, q1)) // line doesn't hit the sphere
		{
			return FALSE;
		}

		if (q1 <= s0 || q0 >= s1) // segments don't overlap
		{
			return FALSE;
		}
		// clip
		if (q0 > s0)
		{
			s0 = q0;
		}
		if (q1 < s1)
		{
			s1 = q1;
		}
		return TRUE;
	}

	template<class VectorType>
	static Int32 RaySphereClipInner(Float64 rad, const VectorType &z, const VectorType &v, Float64 &s0, Float64 &s1, Float64 &t0, Float64 &t1)
	{
		Float64 q0, q1;
		if (!RaySphereIntersection(rad, z, v, q0, q1)) // line doesn't hit the sphere
		{
			return 1;
		}

		// clip linesegment inside the sphere
		if (q1 <= s0 || q0 >= s1) // segments don't overlap
		{
			// q0|//////////|q1  s0------s1
			// s0------s1   q0|////////////|q1
			return 1;
		}
		if (s0 > q0 && s1 < q1) // segment is inside the sphere
		{
			return 0;
		}
		if (s0 < q0)
		{
			// s0---q0|-/-/-/(s1)-/-/-/-|q1---s1
			t0 = q1;
			t1 = s1;
			s1 = q0;
			return t1 > t0 ? 2 : 1;
		}
		else if (s0 < q1)
		{
			// ----q0|-/-/-/-/-s0-/-/-/-|q1----s1
			s0 = q1;
			return 1;
		}
		return 0;
	}


	float ExpApproximation(float x)
	{
		// Should compile to ca 10 instructions.
		// And it can be inlined in contrast to std::exp().
		static_assert(std::numeric_limits<float>::is_iec559, "bah");
		static_assert(sizeof(float) == sizeof(std::uint32_t), "wrong size");

		constexpr float log2_e = 1.4426950408889634f;
		constexpr float poly_coeffs[3] = { 0.34271437f, 0.6496069f , 1.f + 0.0036554f };
		constexpr uint32_t exp_mask = 255u << 23u;

		x *= log2_e;

		const float floored = maxon::Floor(x);
		const int xi = int(floored);
		const float xf = x - floored;
		
		// Overflow and underflow
		if (xi > 128)
			return std::numeric_limits<float>::infinity();

		if (xi < -127)
			return 0.f;

		const float mantisse = xf * (xf*poly_coeffs[0] + poly_coeffs[1]) + poly_coeffs[2];

		std::uint32_t int_view;
		std::memcpy(&int_view, &mantisse, sizeof(float));

		int_view = (int_view & ~exp_mask) | ((((xi + 127)) << 23) & exp_mask);

		float result;
		std::memcpy(&result, &int_view, sizeof(float));

		assert(IsFinite(result));
		return result;
	}

	Float64 ExpApproximation(Float64 x)
	{
		return ExpApproximation((float)x);
	}

	Vector64 ExpApproximation(Vector64 x)
	{
		return Vector64{
			ExpApproximation(x[0]),
			ExpApproximation(x[1]),
			ExpApproximation(x[2]) };
	}
}