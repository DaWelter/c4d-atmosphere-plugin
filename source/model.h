#pragma once

#include "c4d_symbols.h"
#include "c4d.h"
#include "maxon/general_math.h"
#include "maxon/lib_math.h"
#include "maxon/algorithms.h"
#include "maxon/nullallocator.h"
#include "maxon/fixedbufferallocator.h"

#include "Mplanetatmosphere.h"

#include "common.h"

namespace atmosphere {




	struct PhasefunctionRayleigh
	{
		const Float64 Calc(const Float64 mu) const
		{
			return (3. / (16.*maxon::PI))*(1. + mu * mu);
		}
	};

	struct PhasefunctionHg
	{
		const Float64 Calc(const Float64 mu) const
		{
			const Float64 g = 0.76;
			return ((1. - g * g)*(1.f + mu * mu)) / ((2. + g * g)*Pow(1. + g * g - 2.*g*mu, 1.5))*(3. / (8.*maxon::PI));
		}
	};


	struct ScaleQuantities
	{
		Float64 units2km;
		Float64 km2units;

		ScaleQuantities(const BaseContainer &b)
		{
			Float64 radius_units = b.GetFloat(RADIUS_UNITS, 100.);
			Float64 radius_km = b.GetFloat(RADIUS_KM, 6360.);
			units2km = radius_km / radius_units;
			km2units = radius_units / radius_km;
		}
	};


	struct Constituent
	{
		Float64 decayHeight;
		Float64 decayScale;
		Vector64 sealevel_scatter_coeff, sealevel_absorption_coeff;
		Float64 cutoff_radius;

		Constituent() : decayHeight(8.), sealevel_scatter_coeff(0.001)
		{
			decayScale = -1. / decayHeight;
			Vector64 c = sealevel_scatter_coeff;
			cutoff_radius = FMax(decayHeight, FMax(1. / c.x, FMax(1. / c.y, 1. / c.z)))*4.0;
		}

		Constituent(Float64 decayHeight,
			Float64 density,
			Vector phaseFactor,
			Vector absorptionFactor)
			: decayHeight(decayHeight)
		{
			decayScale = -1. / decayHeight;
			sealevel_scatter_coeff = 0.001*density*phaseFactor; // Units of 1/length
			sealevel_absorption_coeff = 0.001*density*absorptionFactor;
			cutoff_radius = (decayHeight*10.0); // + FMax(1./c.x,FMax(1./c.y,1./c.z)));
		}
	};


	// TODO: Fast exponential approximation!
	class ExponentialConstituentModel
	{
		Constituent molecules;
		Constituent aerosoles;
		PhasefunctionRayleigh phasefunction_molecules;
		PhasefunctionHg phasefunction_aerosoles;
		friend class SegmentTransmittanceAnalyticSphere;
		friend class SegmentTransmittanceAnalyticFlat;
		friend class SegmentTransmittanceTabulatedSphere;
		friend class SegmentTransmittanceTabulatedCylinder;
	public:
		Float64 cutoff_radius;
		Float64 lower_cutoff;

		void Init(const BaseContainer &b, const ScaleQuantities &scaling);

		// sum over constituents of phasefunction x sigma_s.
		Vector64 CalcScatteringKernel(double h, const Vector64& v, const Vector64 &u) const
		{
			const Float64 scatter_cos = Dot(v, u);
			Vector64 result{ 0. };
			{
				const Float64 scale_height1 = FMin(aerosoles.decayScale*h, lower_cutoff);
				const Float64 phase_factor1 = phasefunction_aerosoles.Calc(scatter_cos);
				result += phase_factor1 * aerosoles.sealevel_scatter_coeff*Exp(scale_height1);
			}
			{
				const Float64 scale_height2 = FMin(molecules.decayScale*h, lower_cutoff);
				const Float64 phase_factor2 = phasefunction_molecules.Calc(scatter_cos);
				result += phase_factor2 * molecules.sealevel_scatter_coeff*Exp(scale_height2);
			}
			return result;
		}

		Vector64 CalcExtinctionCoefficient(double h) const
		{
			const Float64 scale_height1 = FMin(aerosoles.decayScale*h, lower_cutoff);
			const Float64 scale_height2 = FMin(molecules.decayScale*h, lower_cutoff);
			return (molecules.sealevel_absorption_coeff + molecules.sealevel_scatter_coeff)*Exp(scale_height2) +
				(aerosoles.sealevel_absorption_coeff + aerosoles.sealevel_scatter_coeff)*Exp(scale_height1);
		}

		MediaCoefficients CalcCoefficients(double h) const
		{
			const Float64 aero_decay = Exp(FMin(aerosoles.decayScale*h, lower_cutoff));
			const Float64 molecule_decay = Exp(FMin(molecules.decayScale*h, lower_cutoff));
			const auto aero_extinction = aerosoles.sealevel_absorption_coeff + aerosoles.sealevel_scatter_coeff;
			const auto molecule_extinction = molecules.sealevel_absorption_coeff + molecules.sealevel_scatter_coeff;
			return {
				molecules.sealevel_scatter_coeff * molecule_decay + aerosoles.sealevel_scatter_coeff * aero_decay,
				molecule_extinction * molecule_decay + aero_extinction * aero_decay
			};
		}
	};





#pragma region GeometrySphere
	/********************************************************
		GeometrySphere
	*********************************************************/

	struct GeometrySphere
	{
		Float64 radius;

		void Init(const BaseContainer &b, const ScaleQuantities &scaling)
		{
			radius = b.GetFloat(RADIUS_UNITS, 100.);
		}

		Float64 CalcHeightKm(const Vector64 &p) const
		{
			return (p.GetLength() - radius);
		}

		Vector64 ComputeLowestPointAlong(const Segment &segment) const
		{
			Vector64 center_to_org = segment.p;
			double t_lowest = -Dot(center_to_org, segment.v); // To planet center
			if (t_lowest > segment.dist) // Looking down, intersection with ground is closer
				t_lowest = segment.dist;
			else if (t_lowest < 0.) // Looking up. So the origin is the lowest.
				t_lowest = 0.;
			Vector64 lowest_point = segment.PointAt(t_lowest);
			return lowest_point;
		}

	};


	Segments ComputeSegmentWhereMediumIs(const Vector64 &p, const Vector64 &v, Float64 distance_, Float64 lower_cutoff, Float64 cutoff_radius, const GeometrySphere &geom)
	{
		Float64  t_start = 0;
		Float64  t_end = distance_;
		Segments ret;
		// Clip Outside
		if (atmosphere::RaySphereClip(geom.radius + cutoff_radius, p, v, t_start, t_end))
		{
			Float64 s_start, s_end;
			const auto n = atmosphere::RaySphereClipInner(Max(0., geom.radius - lower_cutoff), p, v, t_start, t_end, s_start, s_end);
			if (n > 0)
			{
				ret[0] = Segment{ p + t_start * v, v, t_end - t_start };
			}
			else
			{
				ret[0] = Segment{ p, v, 0. };
			}
		}
		else
		{
			ret[0] = Segment{ p, v, 0. };
		}
		return ret;
	}

#pragma endregion 

#pragma region GeometryFlat
	/********************************************************
		GeometryFlat
	*********************************************************/

	struct GeometryFlat
	{
		void Init(const BaseContainer &b, const ScaleQuantities &scaling)
		{
		}

		Float64 CalcHeightKm(const Vector64 &p) const
		{
			return p[1];
		}

		Vector64 ComputeLowestPointAlong(const Segment &segment) const
		{
			assert(false);
			return Vector64{};
		}
	};


	Segments ComputeSegmentWhereMediumIs(const Vector64 &p, const Vector64 &v, Float64 distance_, Float64 lower_cutoff, Float64 cutoff_radius, const GeometryFlat &geom)
	{
		// p + t*v = H ?
		const Float64 h = p[1];
		const Float64 mu = v[1];
		const Float64 t0 = (-lower_cutoff - h) / mu;
		const Float64 t1 = (cutoff_radius - h) / mu;
		Segments ret;
		if (h >= cutoff_radius)
		{
			if (mu >= 0.)
			{
				ret[0] = { p, v, 0. };
			}
			else
			{
				assert(t0 >= t1 && t0 >= 0.);
				const Vector64 newp = p + t1 * v;
				const Float dist_above_ground = Min(Max(0., distance_ - t1), t0);
				ret[0] = { newp, v, dist_above_ground };
			}
		}
		else if (h > -lower_cutoff)
		{
			if (mu >= 0.)
			{
				assert(t1 >= 0.);
				ret[0] = { p, v, Min(distance_, t1) };
			}
			else
			{
				assert(t0 >= 0.);
				ret[0] = { p, v, Min(distance_, t0) };
			}
		}
		else
		{
			if (mu > 0.)
			{
				const Vector64 newp = p + t0 * v;
				const Float dist_above_ground = Min(Max(0., distance_ - t0), t1);
				ret[0] = { newp, v, dist_above_ground };
			}
			else
			{
				ret[0] = { p, v, 0. };
			}
		}
		return ret;
	}
#pragma endregion 


#pragma region GeometryCylinder
	/********************************************************
		GeometryCylinder
	*********************************************************/

	struct GeometryCylinder
	{
		Float64 radius;

		void Init(const BaseContainer &b, const ScaleQuantities &scaling)
		{
			radius = b.GetFloat(RADIUS_UNITS, 100.);
		}

		Float64 CalcHeightKm(const Vector64 &p) const
		{
			Vector2d q{ p[0], p[1] };
			return radius - q.GetLength();
		}

		Vector64 ComputeLowestPointAlong(const Segment &segment) const
		{
			assert(false);
			return Vector64{};
		}
	};



	Segments ComputeSegmentWhereMediumIs(const Vector64 &p, const Vector64 &v, Float64 distance_, Float64 lower_cutoff, Float64 cutoff_radius, const GeometryCylinder &geom)
	{
		Segments ret;
		Float64 t_start = 0.;
		Float64 t_end = distance_;
		const Vector2d q{ p[0], p[1] };
		const Vector2d w{ v[0], v[1] };
		if (atmosphere::RaySphereClip<Vector2d>(geom.radius + lower_cutoff, q, w, t_start, t_end)) // Clip ray in outer area
		{
			if (geom.radius < cutoff_radius)
			{
				ret[0] = { p + t_start*v, v, t_end-t_start };
				
				//const double h0 = Vector2d{ ret[0].p.x, ret[0].p.y }.GetLength();
				//const double h1 = XY(ret[0].PointAt(ret[0].dist)).GetLength();
				//assert(h0*1.001 >= geom.radius - cutoff_radius);
				//assert(h1*1.001 >= geom.radius - cutoff_radius);
			}
			else
			{
				Float64  s_start = 0;
				Float64  s_end = 0;
				const auto n = atmosphere::RaySphereClipInner<Vector2d>(geom.radius - cutoff_radius, q, w, t_start, t_end, s_start, s_end);
				if (n >= 1)
				{
					assert(t_end > t_start);
					ret[0] = { p + t_start * v, v, t_end - t_start };
					// Because Cinema ist strange and gives me rays to distances like 10^20.
					ret[0].dist = Min(ret[0].dist, geom.radius*1000.);
					
					//const double h0 = Vector2d{ ret[0].p.x, ret[0].p.y }.GetLength();
					//const double h1 = XY(ret[0].PointAt(ret[0].dist)).GetLength();
					//assert(h0*1.001 >= geom.radius - cutoff_radius);
					//assert(h1*1.001 >= geom.radius - cutoff_radius);
				}
				if (n >= 2)
				{
					assert(s_end > s_start);
					ret[1] = { p + s_start * v, v, s_end - s_start };
					ret[1].dist = Min(ret[1].dist, geom.radius*1000.);
					
					//const double h0 = Vector2d{ ret[1].p.x, ret[1].p.y }.GetLength();
					//const double h1 = XY(ret[1].PointAt(ret[1].dist)).GetLength();
					//assert(h0*1.001 >= geom.radius - cutoff_radius);
					//assert(h1*1.001 >= geom.radius - cutoff_radius);
				}
			}
		}
		return ret;
	}

#pragma endregion


	class ISegmentTransmittance
	{
	public:
		virtual ~ISegmentTransmittance() {}
		virtual Vector64 Evaluate(Float64 t) const = 0;
		virtual Vector64 EvaluateEnd() const = 0;
		virtual maxon::Tuple<Vector64, Float64> BisectTransmittance(Float64 val, Int32 wavelength) const = 0;
	};


#pragma region Model

	class SegmentTransmittanceAnalyticSphere : public ISegmentTransmittance
	{
		// optical depth for ray (r,mu) of length d, using analytic formula
		// (mu=cos(view zenith angle)), intersections with ground ignored
		// H=height scale of exponential density function
		static Float64 DensityIntegral(Float64 radius, Float64 H, Float64 r, Float64 mu, Float64 d)
		{
			const Float64 a = Sqrt((0.5 / H)*r);

			const Float64 a01_x = a * mu, a01_y = a * (mu + d / r);
			const Float64 a01s_x = maxon::SignT(a01_x), a01s_y = maxon::SignT(a01_y);
			const Float64 a01sq_x = a01_x * a01_x, a01sq_y = a01_y * a01_y;
			const Float64 x = a01s_y > a01s_x ? ExpApproximation(a01sq_x) : 0.0;

			const Float64 y_x = a01s_x / (2.3193*Abs(a01_x) + Sqrt(1.52*a01sq_x + 4.0));
			const Float64 y_y = a01s_y / (2.3193*Abs(a01_y) + Sqrt(1.52*a01sq_y + 4.0)) * ExpApproximation(-d / H * (d / (2.0*r) + mu));

			const Float64 val = Sqrt((6.2831*H)*r) * ExpApproximation(FMin((radius - r) / H, 0.)) * (x + y_x - y_y);
			return val;
		}

		const ExponentialConstituentModel* model;
		const GeometrySphere* geom;
		Segment segment;
		double bisect_eps = 1.;
	public:
		struct Parameters
		{
			maxon::ResultMem  Init(const BaseContainer &b, const ScaleQuantities &scale, const ExponentialConstituentModel&, const GeometrySphere&, const InitRenderStruct &) { return maxon::OK;  }
		};

		SegmentTransmittanceAnalyticSphere(const ExponentialConstituentModel& model, const GeometrySphere& geom, const Segment &segment, const Parameters &, VolumeData &)
			: model{ &model }, geom{ &geom }, segment{ segment }
		{
			// Use the shortest mean free path length
			bisect_eps =
				Max((model.molecules.sealevel_absorption_coeff + model.aerosoles.sealevel_scatter_coeff).GetMax(),
				(model.aerosoles.sealevel_absorption_coeff + model.molecules.sealevel_scatter_coeff).GetMax());
			bisect_eps *= 0.1;
		}

		Vector64 OpticalDepth(Float64 t) const
		{
			auto CalcComponentOpticalDepth = [this, t](const Constituent &c) -> Vector64
			{
				Vector64 zenit = segment.p;

				Float64 r = zenit.GetLength();
				zenit /= r;

				const Float64 depth = DensityIntegral(geom->radius, c.decayHeight, r, Dot(zenit, segment.v), t);
				return depth * (c.sealevel_absorption_coeff + c.sealevel_scatter_coeff);
			};
			const Vector64 d1 = CalcComponentOpticalDepth(model->aerosoles);
			const Vector64 d2 = CalcComponentOpticalDepth(model->molecules);
			return d1 + d2;
		}

		Vector64 Evaluate(Float64 t) const override
		{
			return ExpApproximation(-OpticalDepth(t));
		}

		Vector64 EvaluateEnd() const override
		{
			return Evaluate(segment.dist);
		}

		maxon::Tuple<Vector64, Float64> BisectTransmittance(Float64 val, Int32 wavelength) const override
		{
			Vector64 d{ 0. };
			val = maxon::Ln(val);
			auto CalcDepth = [this, wavelength, val, &d](Float64 t) -> Float64
			{
				d = this->OpticalDepth(t);
				return d[wavelength] + val;
			};

			// y = Exp(-d(t))
			// -> -ln(y) = d(t)
			// -> 0. = d(t) + ln(y)

			Float64 t = numerics::bisect_root(0., segment.dist, CalcDepth, bisect_eps, 0.01);
			return { ExpApproximation(-d), t };
		}
	};


	class SegmentTransmittanceAnalyticFlat : public ISegmentTransmittance
	{
		static Float64 DensityIntegral(Float64 H, Float64 h, Float64 mu, Float64 d)
		{
			const Float64 e1 = Exp(-(h + d * mu) / H);
			if (Abs(mu) > 1.e-6)
			{
				const Float64 e2 = Exp(d*mu / H);
				const Float64 result = H / mu * e1 * (e2 - 1.);
				return result;
			}
			else
			{
				const Float64 t1 = d * (1. + 0.5*d * mu / H);
				const Float64 result = t1 * e1;
				return result;
			}
		}

		const ExponentialConstituentModel* model;
		const GeometryFlat* geom;
		Segment segment;
	public:
		struct Parameters
		{
			maxon::ResultMem  Init(const BaseContainer &b, const ScaleQuantities &scale, const ExponentialConstituentModel&, const GeometryFlat&, const InitRenderStruct &) { return maxon::OK;  }
		};

		SegmentTransmittanceAnalyticFlat(const ExponentialConstituentModel& model, const GeometryFlat& geom, const Segment &segment, const Parameters &, VolumeData &)
			: model{ &model }, geom{ &geom }, segment{ segment }
		{
		}

		Vector64 Evaluate(Float64 t) const override
		{
			Float64 h = segment.p[1];
			Float64 mu = segment.v[1];

			auto CalcComponentOpticalDepth = [this, t, h, mu](const Constituent &c) -> Vector64
			{
				Float64 depth = DensityIntegral(c.decayHeight, h, mu, t);
				return depth * (c.sealevel_absorption_coeff + c.sealevel_scatter_coeff);
			};
			const Vector64 d1 = CalcComponentOpticalDepth(model->aerosoles);
			const Vector64 d2 = CalcComponentOpticalDepth(model->molecules);
			return ExpApproximation(-d1 - d2);
		}

		Vector64 EvaluateEnd() const override
		{
			return Evaluate(segment.dist);
		}

		maxon::Tuple<Vector64, Float64> BisectTransmittance(Float64 val, Int32 wavelength) const override
		{
			assert(false); // Not implemented
			return { Vector64{}, 0. };
		}
	};


	template<class Constituents, class Geometry>
	class SegmentTransmittanceNumerical : public ISegmentTransmittance
	{
		static constexpr int BUFFER_SIZE = 1024;
		Vector64 values[BUFFER_SIZE];
		Float64 coords[BUFFER_SIZE];
		int count = 0;
	public:
		struct Parameters
		{
			maxon::ResultMem  Init(const BaseContainer &b, const ScaleQuantities &scale, const Constituents&, const Geometry&, const InitRenderStruct &)
			{
				//num_steps = b.GetInt32(NUM_INTEGRATION_STEPS, 10);
				step_size = b.GetFloat(INTEGRATION_STEP_SIZE, 1.)*scale.km2units;
				//eps = b.GetFloat(MAX_INTEGRATION_ERROR, 0.01);
				//max_level = b.GetInt32(MAX_SUBSTEP_LEVELS, 4);
				return maxon::OK;
			}
			//Int32 num_steps = 10;
			Float64 step_size = 1.;
			Float64 eps = 0.01;
			Int32 max_level = 10;
		};

		SegmentTransmittanceNumerical(const Constituents& model, const Geometry& geom, const Segment &segment, const Parameters &params, VolumeData &vd)
		{
			int num_evals = 0;
			auto result = numerical_integrator::simpson_integration<Vector64>(
				//0., segment.dist, segment.dist*0.1, 0.01, 3,
				0., segment.dist, Min(params.step_size*0.1, segment.dist*0.1), params.eps, params.max_level,
				numerical_integrator::Buffer<Vector64>{coords, values, BUFFER_SIZE},
				[this, model, geom, segment, &vd, &num_evals](Float64 x) -> Vector64
			{
				const double h = geom.CalcHeightKm(segment.p + x * segment.v);
				return model.CalcExtinctionCoefficient(h);
			});
			count = result.num_points;
		}

		Vector64 Evaluate(Float64 t) const override
		{
#if 1
			// TODO: Improve this
			for (int i = 0; i < count - 1; ++i)
			{
				if (coords[i] <= t && coords[i + 1] > t)
				{
					const Float64 f = (t - coords[i]) / (coords[i + 1] - coords[i]);
					const Vector64 depth = values[i + 1] * f + (1. - f)*values[i];
					return ExpApproximation(-depth);
				}
			}
			return ExpApproximation(-values[count - 1]);
#else
			assert(t >= coords[0]);
			const auto item = maxon::LowerBound(coords, coords + count, t);
			if (item == coords)
				return values[0];
			if (item >= coords + count)
				return values[count - 1];
			const Int64 i = (item - coords) - 1;
			const Float64 f = (t - coords[i]) / (coords[i + 1] - coords[i]);
			return values[i + 1] * f + (1. - f)*values[i];
#endif
		}

		Vector64 EvaluateEnd() const override
		{
			return ExpApproximation(-values[count - 1]);
		}

		maxon::Tuple<Vector64, Float64> BisectTransmittance(Float64 val, Int32 wavelength) const override
		{
			assert(false); // Not implemented
			return { Vector64{}, 0. };
		}
	};


	static constexpr UInt64 MAX_SEGMENT_CLASS_SIZE = shit::MaxElement<
		sizeof(SegmentTransmittanceAnalyticSphere),
		sizeof(SegmentTransmittanceAnalyticFlat),
		sizeof(SegmentTransmittanceNumerical<ExponentialConstituentModel, GeometrySphere>),
		sizeof(SegmentTransmittanceNumerical<ExponentialConstituentModel, GeometryCylinder>)
	>();


	template<class Base, UInt64 MAX_SIZE>
	class SinglePolymorphicObjectStaticStorage
	{
		Base* p = nullptr;
		maxon::AlignedStorage< MAX_SIZE, 16> storage;
	public:
		template<class U, class... Args>
		Base* Alloc(Args&&... args)
		{
			assert(sizeof(U) <= MAX_SIZE);
			assert(alignof(U) <= 16);
			p = new (storage._data) U(std::forward<Args>(args)...);
			return p;
		}

		~SinglePolymorphicObjectStaticStorage()
		{
			p->~Base();
		}
	};

	using SegmentTransmittanceHolder = SinglePolymorphicObjectStaticStorage<ISegmentTransmittance, MAX_SEGMENT_CLASS_SIZE>;


	template<class Constituents, class Geometry>
	struct ModelToSegmentTransmittance 
	{ 
		using Type = SegmentTransmittanceNumerical<Constituents, Geometry>; 
	};
	
	template<>
	struct ModelToSegmentTransmittance<ExponentialConstituentModel, GeometrySphere>
	{
		using Type = SegmentTransmittanceAnalyticSphere;
	};
	
	template<>
	struct ModelToSegmentTransmittance<ExponentialConstituentModel, GeometryCylinder>
	{
		using Type = SegmentTransmittanceNumerical<ExponentialConstituentModel, GeometryCylinder>;
	};

	template<>
	struct ModelToSegmentTransmittance<ExponentialConstituentModel, GeometryFlat> 
	{ 
		using Type = SegmentTransmittanceAnalyticFlat; 
	};


	class IAtmosphericModel
	{
	public:
		virtual ~IAtmosphericModel() {}
		virtual maxon::ResultMem Init(const BaseContainer &b, const ScaleQuantities &scaling, const InitRenderStruct &irs) = 0;
		virtual ISegmentTransmittance* GenerateTransmissionModel(SegmentTransmittanceHolder &storage, const Segment &segment, VolumeData &vd) const = 0;
		virtual Vector64 CalcScatteringKernel(const Vector64 &p, const Vector64& v, const Vector64 &u) const = 0;
		virtual Segments ComputeSegmentWhereMediumIs(const Vector64 &p, const Vector64 &v, Float64 distance_) const = 0;
	};


	template<class Constituents, class Geometry>
	class AtmosphericModelBase : public IAtmosphericModel
	{
		using TransmittanceModel = typename ModelToSegmentTransmittance<Constituents, Geometry>::Type;
		Constituents constituents;
		Geometry geometry;
		typename TransmittanceModel::Parameters params;
	public:
		maxon::ResultMem Init(const BaseContainer &b, const ScaleQuantities &scaling, const InitRenderStruct &irs) override
		{
			constituents.Init(b, scaling);
			geometry.Init(b, scaling);
			return params.Init(b, scaling, constituents, geometry, irs);
		}

		ISegmentTransmittance* GenerateTransmissionModel(SegmentTransmittanceHolder &storage, const Segment &segment, VolumeData &vd) const override
		{
			return storage.Alloc<TransmittanceModel>(constituents, geometry, segment, params, vd);
		}

		Vector64 CalcScatteringKernel(const Vector64 &p, const Vector64& v, const Vector64 &u) const override
		{
			const Float64 h = geometry.CalcHeightKm(p);
			return constituents.CalcScatteringKernel(h, v, u);
		}

		Segments ComputeSegmentWhereMediumIs(const Vector64 &p, const Vector64 &v, Float64 distance_) const override
		{
			return atmosphere::ComputeSegmentWhereMediumIs(p, v, distance_, constituents.lower_cutoff, constituents.cutoff_radius, geometry);
		}
	};

	static constexpr UInt64 MAX_ATMOSPHERIC_MODEL_CLASS_SIZE = shit::MaxElement<
		sizeof(AtmosphericModelBase<ExponentialConstituentModel, GeometrySphere>),
		sizeof(AtmosphericModelBase<ExponentialConstituentModel, GeometryFlat>),
		sizeof(AtmosphericModelBase<ExponentialConstituentModel, GeometryCylinder>)
	>();

	using AtmosphericModelHolder = SinglePolymorphicObjectStaticStorage<IAtmosphericModel, MAX_ATMOSPHERIC_MODEL_CLASS_SIZE>;

#pragma endregion

}