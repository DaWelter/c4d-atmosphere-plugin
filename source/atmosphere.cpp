#include "c4d_symbols.h"
#include "c4d.h"

#include "customgui_matpreview.h"
#include "customgui_inexclude.h"

#include "maxon/general_math.h"
#include "maxon/lib_math.h"
#include "maxon/algorithms.h"
#include "maxon/nullallocator.h"
#include "maxon/fixedbufferallocator.h"
#include "maxon/range.h"
#include "c4d_quaternion.h"

#include "Mplanetatmosphere.h"
//#include "osphere.h"

#include "common.h"
#include "integrator.h"
#include "model.h"

#include <array>



#define ID_ATMOSPHERE2	1029487

/*
https://en.wikipedia.org/wiki/Adaptive_quadrature

Fejér Quadrature
https://chaospy.readthedocs.io/en/master/quadrature.html

https://git.savannah.gnu.org/cgit/gsl.git/tree/integration/qag.c#n41
https://git.savannah.gnu.org/cgit/gsl.git/tree/integration/qk.c

https://www.sciencedirect.com/science/article/pii/037704279190201T

https://github.com/stevengj/cubature/blob/master/hcubature.c
*/

namespace numerical_integrator
{

	template<>
	double norm<Vector64>(const Vector64 &x)
	{
		return Max(Abs(x[0]), Max(Abs(x[1]), Abs(x[2])));
	}

	template<>
	Vector64 zero<Vector64>()
	{
		return Vector64{};
	}

	template<>
	inline bool all_abs_less_eq(const Vector64& a, double b_mult, const Vector64& b)
	{
#if (API_VERSION < 2023000)
    {
      return Abs(a).LessThanOrEqual(b_mult * Abs(b));
    }
#else
    {
      return Abs(a) <= (b_mult * Abs(b));
    }
#endif
	}
}





namespace atmosphere {
//using maxon::Dot;
//using maxon::Cross;
using maxon::Exp;
using maxon::BaseArray;


static Int32 ConstitId(Int32 id, Int32 grp) { return (grp * 1000 + 2000) + id; }


Int32 GetParameterInt32(GeListNode* node, DescID id)
{
	GeData param;
	node->GetParameter(id, param, DESCFLAGS_GET::NONE);
	return param.GetInt32();
}


void ExponentialConstituentModel::Init(const BaseContainer &b, const ScaleQuantities &scaling)
{
	Float global_density = b.GetFloat(GLOBAL_DENSITY, 1.);
	auto InitConstituent = [&](Constituent &c, Int32 grp)
	{
		Bool on = b.GetBool(ConstitId(ID_ON, grp), FALSE);
		Float density = on ? b.GetFloat(ConstitId(ID_DENSITY, grp), 1.) : 0.;
		c = Constituent(
			scaling.km2units*b.GetFloat(ConstitId(ID_DECAY_HEIGHT, grp), 8.),
			global_density*density,
			scaling.units2km*b.GetVector(ConstitId(ID_PHASE_FACTOR, grp), Vector(5.8, 13.5, 33.1)),
			scaling.units2km*b.GetVector(ConstitId(ID_ABSORPTION_FACTOR, grp), Vector(0.))
		);
	};

	InitConstituent(molecules, 0);
	InitConstituent(aerosoles, 1);
	cutoff_radius = Max(molecules.cutoff_radius, aerosoles.cutoff_radius);
	lower_cutoff = 0.1*cutoff_radius;
}


BaseObject* FindObject(BaseObject* first, const maxon::String &name)
{
	while (first)
	{
		if (first->GetName() == name)
			return first;
		auto *result = FindObject(first->GetDown(), name);
		if (result)
			return result;
		first = first->GetNext();
	}
	return nullptr;
}





//#define TEST_INTEGRATION


// be sure to use a unique ID obtained from www.plugincafe.com
#define ID_PARTICLEVOLUME	1024852


#pragma region Plugin

struct MyRender
{
public:
  MyRender(MyRender &&) = delete; // Because of the mutex!
  MyRender& operator=(MyRender &&) = delete;
  MyRender(const MyRender &) = delete;
  MyRender& operator=(const MyRender &) = delete;

  MyRender(const BaseContainer &b, const InitRenderStruct &is_, BaseMaterial* mat_, const BaseArray<const RayLight*> &light_incl_, const TexData *tex)
    : my_tex_data{ tex }
  {
    light_incl = light_incl_.GetFirst();
    light_incl_count = light_incl_.GetCount();

    Matrix64 m = tex->m;
    m.sqmat.Normalize();
    this->local_frame = m;
    this->local_frame_inv = ~m;

#ifdef USE_RADIANCE_CACHE
	maxon::Range<Vector64> bounds = atmospheric_model.GetBounds();
    this->max_cache_radius = 200.0 * scaling.km2units;
    this->min_cache_spacing = 2. * scaling.km2units;

    for (Int32 i = 0; i < is_.vd->GetCPUCount(); ++i)
    {
      threadlocal.Append(ThreadLocal{});
      threadlocal[i].radiance_cache.Init(Vector32(bounds.GetCenter()), bounds.GetDimension().GetMax());
    }
    shared.radiance_cache.Init(Vector32(bounds.GetCenter()), bounds.GetDimension().GetMax());
#endif
  }

	maxon::ResultMem Init(const BaseContainer& b, const InitRenderStruct& is_)
	{
		iferr_scope_handler
		{
			return maxon::FAILED;
		};

		ScaleQuantities scaling(b);

		switch (b.GetInt32(GEOMETRY_TYPE))
		{
		case ID_GEOMETRY_FLAT:
			atmospheric_model = model_storage.Alloc<AtmosphericModelBase<ExponentialConstituentModel, GeometryFlat>>();
			break;
		case ID_GEOMETRY_CYLINDER:
			atmospheric_model = model_storage.Alloc<AtmosphericModelBase<ExponentialConstituentModel, GeometryCylinder>>();
			break;
		default:
			atmospheric_model = model_storage.Alloc<AtmosphericModelBase<ExponentialConstituentModel, GeometrySphere>>();
			break;
		}

		scatter_intensity_factor = b.GetFloat(SCATTER_INTENSITY, 1.);
		integration_step_size = b.GetFloat(INTEGRATION_STEP_SIZE, 1.)*scaling.km2units;
		integration_eps = b.GetFloat(MAX_INTEGRATION_ERROR, 0.01);
		integration_eps_low_accuracy = b.GetFloat(MAX_INTEGRATION_ERROR_FOR_GI, 1.);
		max_integration_levels = 10; //b.GetInt32(MAX_SUBSTEP_LEVELS, 4);
		use_monte_carlo = b.GetBool(MONTE_CARLO_HYBRID_INTEGRATION, false);

		atmospheric_model->Init(b, scaling, is_) iferr_return;

		randgens.Resize(is_.vd->GetCPUCount()) iferr_return;
		UInt32 seed = 1234567;
		for (auto& gen : randgens)
		{
			gen.Init(seed);
			seed = seed * 5464759 + 10985; // Just rolled my face over KB.
		}
		return maxon::OK;
	}


private:
	AtmosphericModelHolder model_storage;
	IAtmosphericModel* atmospheric_model;
	mutable BaseArray<Random> randgens;
	
	TexData const * const my_tex_data;

	typedef const RayLight* RayLightP;
	const RayLightP* light_incl;

	Matrix64 local_frame; // No scaling allowed
	Matrix64 local_frame_inv;

	Float64 scatter_intensity_factor;
	Float64 integration_step_size;
	Float64 integration_eps;
	Float64 integration_eps_low_accuracy;
	Int32 max_integration_levels;
	Int light_incl_count;  
	Bool use_monte_carlo = false;
  
#ifdef USE_RADIANCE_CACHE
  struct CacheRecord
  {
    Vector32 position;
    Vector32 irradiance;
    maxon::SqMat3<Vector32> gradient;
    Float32 radius_vertical;
    Float32 radius_horizontal;
  };

  // 
  struct ThreadLocal
  {
    maxon::BaseArray<CacheRecord> cache_records;
    Octree::DynamicLooseOctree radiance_cache;
  };

  struct CacheQueryPosition
  {
    Vector32 position;
    Vector32 up; // The Up-Vector, w.r.t to the sky. Normalized.
  };

  ThreadLocal shared;
  BaseArray<ThreadLocal> threadlocal;
  //maxon::Spinlock mutex;

  Float32 max_cache_radius = 0;
  Float32 min_cache_spacing = 0;
  static constexpr Float32 radius_eps = 0.1f;
  static constexpr Int32 MAX_NUM_USED_RECORDS = 32;
  static constexpr Int32 REQUIRED_NUM_RECORDS = 1;
#endif

private:
  Bool IlluminateAnyPoint(VolumeData* vd, const RayLight* rl, Vector *col, Vector64 *lv, const Vector64 &p, ILLUMINATEFLAGS flags1, RAYBIT flags2) const
  {      
    Bool nodif=FALSE,nospec=FALSE;
    if (rl->lr.object) CalcRestrictionInc(&rl->lr,vd->op,nodif,nospec);
    if(nodif) return FALSE;
    if(light_incl_count>0) 
    {
      Bool found;
      Int i = light_incl_count-1;
      do 
      {
        found = rl==light_incl[i--];
      } while(i>=0 && !found);
      if(!found) return FALSE;
    }
    auto world_p = local_frame * p;
    Bool ok = vd->IlluminateAnyPoint(rl,col,lv,world_p,flags1,flags2);
    *lv = local_frame_inv.sqmat * (*lv);
    return ok;
  }
  
#ifdef USE_RADIANCE_CACHE
  static Float32 FractionalDistanceFromRecord(const CacheRecord &cr, const CacheQueryPosition &qp)
  {
    auto delta = cr.position - qp.position;
    auto d2_vert_horiz = ProjectGetSquaredParallelPerpendicular(qp.up, delta);
    // d_v^2 / r_v^2 + d_h^2 / r_h^2 <= 1.  i.e. delta lies within an elipsoid.
    auto val = d2_vert_horiz.Get<0>() / (cr.radius_vertical * cr.radius_vertical) + d2_vert_horiz.Get<1>() / (cr.radius_horizontal * cr.radius_horizontal);
    return val;
  }


  static Bool IsCacheRecordValidForQuery(const CacheRecord &cr, const SegmentwiseCalculations &segmentwise, const CacheQueryPosition &qp)
  {
    auto val = FractionalDistanceFromRecord(cr, qp);
    return val < 1.f;
  }
  

  Int32 FindValidCacheRecordsForQuery(const ThreadLocal& tl, const SegmentwiseCalculations &segmentwise, const CacheQueryPosition &qp, CacheRecord *result) const
  {
    Int32 num_used_records = 0;
    tl.radiance_cache.PointQuery(qp.position, [&](const int32_t *indices, std::size_t n) {
      while (n > 0 && num_used_records < MAX_NUM_USED_RECORDS)
      {
        const auto & cr = tl.cache_records[indices[--n]];
        if (IsCacheRecordValidForQuery(cr, segmentwise, qp))
        {
          result[num_used_records++] = cr;
        }
      }
    });
    return num_used_records;
  }

  Bool IsAdmissibleForInsertion(const ThreadLocal& tl, const Vector32 &p) const
  {
    Bool res = true;
    tl.radiance_cache.PointQuery(p, [&](const int32_t *indices, std::size_t n) {
      while (n-- > 0 && res)
      {
        const auto i = indices[n];
        if ((tl.cache_records[i].position - p).GetSquaredLength() < min_cache_spacing*min_cache_spacing)
          res = false;
      }
    });
    return res;
  }

  void InsertRecord(ThreadLocal& tl, const CacheRecord &cr, VolumeData* vd)
  {
    const auto cache_ok = tl.cache_records.Append(cr);
    if (cache_ok == maxon::OK)
    {
      const auto insert_ok = tl.radiance_cache.InsertSphere(tl.cache_records.GetCount() - 1, cr.position, maxon::Max(cr.radius_horizontal, cr.radius_vertical));
      if (!(insert_ok == maxon::OK))
      {
        tl.cache_records.Pop();
        vd->OutOfMemory();
      }
    }
    else
      vd->OutOfMemory();
  }


  Vector64 InterpolateRecordRadiance(const CacheRecord *records, Int32 count, const CacheQueryPosition &qp) const
  {
    Float32 weight_sum = 0;
    Float32 weights[MAX_NUM_USED_RECORDS];
    for (Int32 i = 0; i < count; ++i)
    {
      auto dist = 1.0f - FractionalDistanceFromRecord(records[i], qp);
      DebugAssert(dist >= -1.e-6);
      weights[i] = (3.f - 2.0f*dist)*dist*dist;
      weight_sum += weights[i];
    }
    auto res = Vector64{ 0 };


    auto GeoExtrapolation = [this](const CacheRecord &r, const CacheQueryPosition &qp)
    {
      auto frame = OrthogonalSystem<2>(qp.up);
      // Altitude difference
      auto query_altitude = Dot(qp.up, qp.position);
      auto rec_altitude = r.position.GetLength();
      // Direction to query point projected to tangent plane
      auto dpx = Dot(frame.v1, qp.position - r.position);
      auto dpy = Dot(frame.v2, qp.position - r.position);
      auto norm_inv = maxon::Sqrt(dpx*dpx + dpy*dpy);
      if (norm_inv > 0.f)
        norm_inv = 1.f / norm_inv;
      // Geodesic distance
      auto geo_dist = 0.5f*(query_altitude+rec_altitude) * maxon::ACos(maxon::Clamp01(Dot(r.position, qp.up) / rec_altitude));
      // Interpolate:
      auto dp = Vector32(dpx*norm_inv*geo_dist, dpy*norm_inv*geo_dist, query_altitude - rec_altitude);
           dp = frame * dp;
      auto extrapolated = r.irradiance
        + r.gradient.v1 * dp.x
        + r.gradient.v2 * dp.y
        + r.gradient.v3 * dp.z;
      //DebugAssert(IsFinite(extrapolated.x) && IsFinite(extrapolated.y) && IsFinite(extrapolated.z));
      return extrapolated;
    };
    
    static constexpr Float32 EPS = 1.e-3;

    auto LogExtrapolation = [this](const CacheRecord &r, const CacheQueryPosition &qp)
    {
      auto dp = qp.position - r.position;
      // grad log x + eps = 1./(x+eps) * grad x
      auto log_irrad = Vector32(Ln(r.irradiance[0] + EPS), Ln(r.irradiance[1] + EPS), Ln(r.irradiance[2] + EPS));
      auto grad = r.gradient;
      grad.v1.x /= r.irradiance[0] + EPS;
      grad.v2.x /= r.irradiance[0] + EPS;
      grad.v3.x /= r.irradiance[0] + EPS;
      grad.v1.y /= r.irradiance[1] + EPS;
      grad.v2.y /= r.irradiance[1] + EPS;
      grad.v3.y /= r.irradiance[1] + EPS;
      grad.v1.z /= r.irradiance[2] + EPS;
      grad.v2.z /= r.irradiance[2] + EPS;
      grad.v3.z /= r.irradiance[2] + EPS;
      auto extrapolated = log_irrad
        + grad.v1 * dp.x
        + grad.v2 * dp.y
        + grad.v3 * dp.z;
      return extrapolated;
    };

    for (Int32 i = 0; i < count; ++i)
    {
      //auto extrapolated = GeoExtrapolation(records[i], qp);
      //auto extrapolated = LogExtrapolation(records[i], qp);
      auto dp = qp.position - records[i].position;
      auto extrapolated = records[i].irradiance
        + records[i].gradient.v1 * dp.x
        + records[i].gradient.v2 * dp.y
        + records[i].gradient.v3 * dp.z;
      res += weights[i]*extrapolated;
    }
    if (count > 0)
      res *= 1.0f / weight_sum;
    //res[0] = !std::isinf(res[0]) ? Exp(res[0]) - EPS : 0.f;
    //res[1] = !std::isinf(res[1]) ? Exp(res[1]) - EPS : 0.f;
    //res[2] = !std::isinf(res[2]) ? Exp(res[2]) - EPS : 0.f;
    DebugAssert(IsFinite(res.x) && IsFinite(res.y) && IsFinite(res.z));
    return res;
  }


  Vector64 CalculateInscatterAndFillCache(ThreadLocal& tl, VolumeData* vd, const SegmentwiseCalculations &segmentwise, Float64 t, const Vector64 &p)
  {
    auto QueryLightsForIrradiance = [&]() ->  maxon::Tuple<Vector32, maxon::SqMat3<Vector32> > // Return value and gradient
    {
      Vector64 radiances[4] = {};
      const Float eps = 1.e-6*(atmospheric_model.molecules.cutoff_radius + atmospheric_model.aerosoles.cutoff_radius);
      const Vector64 dp[4] = {
        Vector64{ 0, 0, 0},
        Vector64{ eps, 0, 0},
        Vector64{ 0, eps, 0},
        Vector64{ 0, 0, eps}
      };
      for (Int32 i = vd->GetLightCount() - 1; i >= 0; --i)
      {
        const RayLight* rl = vd->GetLight(i);
        Vector col;
        Vector64 lv; // Points to the light
        const RayObject* rop = vd->op;
        vd->op = NULL;    // remove the object link so that inscatter rays can be computed for objects that are excluded from actual illumination.  
        const auto flags = ILLUMINATEFLAGS::SHADOW | ILLUMINATEFLAGS::DISABLESHADOWMAP_CORRECTION;

        for (Int32 delta_p_idx = 0; delta_p_idx < 4; ++delta_p_idx)
        {
          const auto actual_p = p + dp[delta_p_idx];
          if (IlluminateAnyPoint(vd, rl, &col, &lv, actual_p, flags, RAYBIT::CURR_CUSTOM | RAYBIT::VOLUMETRICLIGHT))
          {
            if (!NearEqual(col, Vector(0.), 1.e-6))
            {
              // 'out' points to the viewer
              // It is not nessesary to compute transmission from space to earth because that is done automatically if shadows are enabled!
              Vector64 scatter_coeff = segmentwise.CalcScatteringCoeffAt(actual_p);
              radiances[delta_p_idx] += scatter_coeff * col;
            }
          }
        }

        vd->op = rop;
      }
      
      const Float64 eps_inv = 1.0 / eps;
      maxon::SqMat3<Vector32> grad;
      grad.v1 = Vector32((radiances[1] - radiances[0])*eps_inv); // dL/dx
      grad.v2 = Vector32((radiances[2] - radiances[0])*eps_inv); // dL/dy
      grad.v3 = Vector32((radiances[3] - radiances[0])*eps_inv); // dL/dz
      return maxon::ToTuple(radiances[0], grad);
    };
    
    auto ComputeRadii = [this](const Vector32 radiance, const maxon::SqMat3<Vector32> grad, const Vector32 up)
    {
      auto transp = grad.GetTransposed(); // v1, v2, v3 now denote wavelengths.
      auto g_r_norm2 = ProjectGetSquaredParallelPerpendicular(up, transp.v1);
      auto g_g_norm2 = ProjectGetSquaredParallelPerpendicular(up, transp.v2);
      auto g_b_norm2 = ProjectGetSquaredParallelPerpendicular(up, transp.v3);
      Vector32 tmp(radiance);
      DivSafeFromNan(tmp.x, maxon::Sqrt(g_r_norm2.Get<0>()));
      DivSafeFromNan(tmp.y, maxon::Sqrt(g_g_norm2.Get<0>()));
      DivSafeFromNan(tmp.z, maxon::Sqrt(g_b_norm2.Get<0>()));
      Float32 rad_vert = maxon::Min<Float32>(max_cache_radius, radius_eps*tmp.GetMin());
      tmp = radiance;
      DivSafeFromNan(tmp.x, maxon::Sqrt(g_r_norm2.Get<1>()));
      DivSafeFromNan(tmp.y, maxon::Sqrt(g_g_norm2.Get<1>()));
      DivSafeFromNan(tmp.z, maxon::Sqrt(g_b_norm2.Get<1>()));
      Float32 rad_horiz = maxon::Min<Float32>(max_cache_radius, radius_eps*tmp.GetMin());
      return maxon::Tuple<Float32,Float32>(rad_vert, rad_horiz);
    };


    Vector64 res;
    Int32 num_used_records = 0;
    CacheRecord used_records[MAX_NUM_USED_RECORDS];
    const CacheQueryPosition qp32{ Vector32(p), Vector32(geometry_model.UpVector(p)) };

    MAXON_SCOPE  // Lock the mutex
    {
      //maxon::ScopedLock lock(this->cache_mutex);
      num_used_records = FindValidCacheRecordsForQuery(tl, segmentwise, qp32, used_records);
    }
    
    if (num_used_records < REQUIRED_NUM_RECORDS)
    {
      if (tl.radiance_cache.IsInBounds(qp32.position))
      {
        auto res = QueryLightsForIrradiance(); // Can take a long time.
        auto radii = ComputeRadii(res.Get<0>(), res.Get<1>(), qp32.up);
        MAXON_SCOPE
        {
          //maxon::ScopedLock lock(this->cache_mutex);
          if (IsAdmissibleForInsertion(tl, qp32.position))
          {
            InsertRecord(tl, CacheRecord{
              qp32.position,
              res.Get<0>(),
              res.Get<1>(),
              radii.Get<0>(),
              radii.Get<1>()
            }, vd);
          }
        }
      }
      else
        res = Vector64{ 0 };
    }
    else
    {
      res = InterpolateRecordRadiance(used_records, num_used_records, qp32);
    }

    return res;
  }


  Vector64 CalculateCachedInscatter(const ThreadLocal& tl, VolumeData* vd, const SegmentwiseCalculations &segmentwise, Float64 t, const CacheQueryPosition &qp) const
  {
    CacheRecord used_records[MAX_NUM_USED_RECORDS];
    Int32 num_used_records = FindValidCacheRecordsForQuery(tl, segmentwise, qp, used_records);
    return InterpolateRecordRadiance(used_records, num_used_records, qp);
  }
#endif

  Vector64 CalculateDirectInscatteredRadianceAtPoint(VolumeData* vd, const Vector64 &v, const Vector64 &p) const
  {        
    Vector64 res(0.);
    for(Int32 i=vd->GetLightCount()-1; i>=0; --i)
    {
      const RayLight* rl = vd->GetLight(i);
      Vector col;
      Vector64 lv; // According to docs: Assigned the light to point vector. For area and tube lights the vector will use the center of the light source. 
      const RayObject* rop = vd->op;
      vd->op = NULL;    // remove the object link so that inscatter rays can be computed for objects that are excluded from actual illumination.  
      const auto flags = ILLUMINATEFLAGS::SHADOW | ILLUMINATEFLAGS::DISABLESHADOWMAP_CORRECTION;
      if(IlluminateAnyPoint(vd,rl,&col,&lv,p, flags, RAYBIT::CURR_CUSTOM))
      {
        if(!NearEqual(col,Vector(0.),1.e-6))
        {
          // 'out' points to the viewer
          // It is not nessesary to compute transmission from space to earth because that is done automatically if shadows are enabled!
          Vector64 scatter_coeff = atmospheric_model->CalcScatteringKernel(p, v, lv);
          res += scatter_coeff * col;
        }
      }
      vd->op = rop;
    }
    return res;
  }

 
  maxon::Tuple<Vector64,Vector64> IntegrateInscatteredRadiance(VolumeData* vd, const Segment &segment) const
  {
	SegmentTransmittanceHolder storage;
	auto* transmittance_model = atmospheric_model->GenerateTransmissionModel(storage, segment, *vd);
	const Float64 r0 = use_monte_carlo ? randgens[vd->GetCurrentCPU()].Get11() : 0.;

	// Global error control is faster than the local control - in spite of the complication with the priority queue.
	// I suppose the lower number of function evaluations needed to reach the error threshold makes more than up for it.

	const Vector64 radiance = numerical_integrator::fejer_global<Vector64, 128>(
		0.,
		segment.dist,
		integration_step_size,
		CheckUseLowerAccuracy(vd) ? integration_eps_low_accuracy :  integration_eps,
		max_integration_levels,
		[this, vd, segment, &transmittance_model](Float64 t) -> Vector64
	{
		if (vd->TestBreak())
			return Vector64{};
		auto tr = transmittance_model->Evaluate(t);
		auto radiance = CalculateDirectInscatteredRadianceAtPoint(vd, segment.v, segment.p + t*segment.v);
		return tr * radiance;
	},
	r0);
	assert(maxon::AllFinite(radiance));
	//const Vector64 radiance{ 0. };
	return { radiance, transmittance_model->EvaluateEnd() };
  }

   
  maxon::Tuple<Vector64, Vector64> IntegrateInscatteredRadiance(VolumeData* vd, const Segments &segments) const
  {
#if 1
	  Vector64 col{ 0. }, trans{ 1. };
	  if (segments[0])
	  {
		  maxon::Tie(col, trans) = IntegrateInscatteredRadiance(vd, segments[0]);
		  if (segments[1])
		  {
			  Vector64 col2{ 0. }, trans2{ 1. };
			  maxon::Tie(col2, trans2) = IntegrateInscatteredRadiance(vd, segments[1]);
			  col = col + trans * col2;
			  trans = trans * trans2; 
		  }
	  }
	  return { col, trans };
#else
	  Float64 r1 = randgens[vd->GetCurrentCPU()].Get01();
	  auto wavelength = Int32(r1 * 2.999999999);
	  Vector64 col{ 0. }, trans{ 1. };

	  for (size_t i = 0; i < segments.size() && segments[i]; ++i)
	  {
		  Float64 r0 = randgens[vd->GetCurrentCPU()].Get01();
		  SegmentTransmittanceHolder storage;
		  auto* transmittance_model = atmospheric_model->GenerateTransmissionModel(storage, segments[i], *vd);
		  Vector64 tr_at_end = transmittance_model->EvaluateEnd();
		  Float64 t; Vector64 tr;
		  maxon::Tie(tr, t) = transmittance_model->BisectTransmittance(1.-r0*(1.-tr_at_end[wavelength]), wavelength);
		  if (IsFinite(t))
		  {
			  auto radiance = CalculateDirectInscatteredRadianceAtPoint(vd, segments[i].v, segments[i].p + t * segments[i].v);
			  auto coeffs = atmospheric_model->EvaluateCoeffs(segments[i].PointAt(t));
			  tr *= trans;
			  Float64 pdf = Div(tr * coeffs.sigma_t, Vector64{ 1. } - tr_at_end).GetAverage();
			  col += (tr * radiance) / pdf;
		  }
		  trans *= transmittance_model->EvaluateEnd();
	  }

	  //trans /= trans.GetAverage();
	  return { col, trans };
#endif
  }

  Vector64  CalcTransmittance(VolumeData* vd, const Segment &segment) const
  {
	  SegmentTransmittanceHolder storage;
	  auto* transmittance_model = atmospheric_model->GenerateTransmissionModel(storage, segment, *vd);
	  return transmittance_model->EvaluateEnd();
  }

  Vector64  CalcTransmittance(VolumeData* vd, const Segments &segments) const
  {
	Vector64 tr = segments[0] ? 
		CalcTransmittance(vd, segments[0]) : 
		Vector64{ 1. };
	if (segments[1])
		tr *= CalcTransmittance(vd, segments[1]);
	return tr;
  }

  static bool CheckUseLowerAccuracy(BaseVolumeData *vd)
  {
	  return /*vd->GetRayParameter()->gi_prepass ||*/ (vd->raybits & RAYBIT::BLURRY) || (vd->raybits & RAYBIT::GI);
  }


public:
  void CalcVolumetric(VolumeData *vd)
  {
    auto p = local_frame_inv * vd->ray->p;
    auto v = local_frame_inv.sqmat * vd->ray->v;
	const auto segments = atmospheric_model->ComputeSegmentWhereMediumIs(p, v, vd->dist);
	maxon::Tie(vd->col, vd->trans) = IntegrateInscatteredRadiance(vd, segments);
	vd->col *= scatter_intensity_factor;
	assert(maxon::AllFinite(vd->col));
	assert(maxon::AllFinite(vd->trans));
  }

  void CalcTransparency(VolumeData *vd)
  {
    auto p = local_frame_inv * vd->ray->p;
    auto v = local_frame_inv.sqmat * vd->ray->v;
	const auto segments = atmospheric_model->ComputeSegmentWhereMediumIs(p, v, vd->dist);
	vd->trans = CalcTransmittance(vd, segments);
	assert(maxon::AllFinite(vd->trans));
  }

  maxon::Result<void> OnRenderingStart(VolumeData* vd)
  {
    iferr_scope;
#ifdef USE_RADIANCE_CACHE
    for (auto &tl : threadlocal)
    {
      for (const CacheRecord &cr : tl.cache_records)
      {
        if (IsAdmissibleForInsertion(shared, cr.position))
          InsertRecord(shared, cr, vd);
      }
    }
#endif
    return maxon::OK;
  }


  void DumpCacheTo(BaseDocument *doc)
  {
#ifdef USE_RADIANCE_CACHE
    auto* parent = BaseObject::Alloc(Onull);
    ObjectColorProperties cp;
    for (const CacheRecord &cr : shared.cache_records)
    {
      auto* o = BaseObject::Alloc(Onull);
      o->SetParameter(DescID(NULLOBJECT_RADIUS), GeData(1.0), DESCFLAGS_SET::NONE);
      Matrix m = Matrix::NullValue(); 
      m.sqmat = OrthogonalSystem<0>(Vector64(cr.position));
      m.off = Vector64(cr.position);
      m.sqmat.v1 *= cr.radius_vertical;
      m.sqmat.v2 *= cr.radius_horizontal;
      m.sqmat.v3 *= cr.radius_horizontal;
      o->SetMg(m);
      o->SetParameter(DescID(NULLOBJECT_DISPLAY), GeData(NULLOBJECT_DISPLAY_CUBE), DESCFLAGS_SET::NONE);
      o->SetParameter(DescID(NULLOBJECT_ORIENTATION), GeData(NULLOBJECT_ORIENTATION_XY), DESCFLAGS_SET::NONE);
      o->InsertUnder(parent);
    }
    doc->InsertObject(parent, nullptr, nullptr);
#endif
  }


  Int64 GetCacheSize() const
  {
#ifdef USE_RADIANCE_CACHE
    return shared.cache_records.GetCount();
#else
	  return 0;
#endif
  }


  // Project x on normalized vector w. Return squared length of parallel and perpendicular components.
  template <typename T, Int S>
  static maxon::Tuple<T, T> ProjectGetSquaredParallelPerpendicular(const maxon::Vec3<T, S>& w, const maxon::Vec3<T, S>& x)
  {
    auto d_vertical = Dot(x, w);
    auto d2_horizontal = (x - d_vertical * w).GetSquaredLength();
    return maxon::Tuple<T, T>(d_vertical*d_vertical, d2_horizontal);
  }

  template <typename T>
  static void DivSafeFromNan(T &a, const T b)
  {
    a = b > 0 ? a / b : std::numeric_limits<T>::infinity();
  }
};


class ParticleVolume;

#if 0
class AtmosphereVideopost : public VideoPostData
{
  INSTANCEOF(AtmosphereVideopost, VideoPostData)
  friend class ParticleVolume;
  BaseArray<ParticleVolume*> shaders;
public:
  virtual Bool Init(GeListNode* node);
  static NodeData* Alloc() { return NewObjClear(AtmosphereVideopost); }

  virtual RENDERRESULT Execute(BaseVideoPost* node, VideoPostStruct* vps);
  virtual VIDEOPOSTINFO GetRenderInfo(BaseVideoPost* node) { return VIDEOPOSTINFO::NONE; }

  virtual Bool RenderEngineCheck(BaseVideoPost* node, Int32 id);
};
#endif


class ParticleVolume : public MaterialData
{
    INSTANCEOF(ParticleVolume,MaterialData)
  private:
    void UpdateGuiValues(BaseMaterial* node);
    BaseArray<maxon::UniqueRef<MyRender>> renders;
    BaseArray<TexData const*> tex_list;  // Texture tags that have this material
    BaseArray<const RayLight*> included_lights;
    enum PreviewSceneId : Int32 {
      PREVIEW_ATMOS_GROUND = 1,
	  PREVIEW_ATMOS_GROUND2 = 2,
      PREVIEW_ATMOS_LOW = 3,
      PREVIEW_ATMOS_HIGH = 4,
    };
	enum PreviewLightType : Int32 {
		PREVIEW_LIGHT_DISTANT = 8,
		PREVIEW_LIGHT_POINT = 16
	};
    PreviewSceneId preview_scene_id = PREVIEW_ATMOS_HIGH;
	PreviewLightType preview_light_type = PREVIEW_LIGHT_DISTANT;
	
	static Int32 MakeMenuNum(PreviewSceneId scene_id, PreviewLightType light_type, Int32 geom_id)
	{
		return scene_id | light_type;
	}
	static maxon::Tuple<PreviewSceneId,PreviewLightType> ExtractPreviewConfigFromMenuNum(Int32 menu_id, Int32 geom_id)
	{
		auto scene_id = static_cast<PreviewSceneId>(menu_id & (8 - 1));
		auto light_type = static_cast<PreviewLightType>(menu_id & (8 | 16));
		return { scene_id, light_type };
	}
	static maxon::String ToString(PreviewSceneId scene_id)
	{
		switch (scene_id)
		{
		case PREVIEW_ATMOS_GROUND: return GeLoadString(IDS_GROUND);
		case PREVIEW_ATMOS_GROUND2: return GeLoadString(IDS_GROUND2);
		case PREVIEW_ATMOS_LOW: return GeLoadString(IDS_LOW);
		case PREVIEW_ATMOS_HIGH: return GeLoadString(IDS_HIGH);
		}
		return ""_s;
	}
	static maxon::String ToString(PreviewLightType light_type)
	{
		switch (light_type)
		{
		case PREVIEW_LIGHT_DISTANT: return GeLoadString(IDS_LIGHT_DIRECTIONAL);
		case PREVIEW_LIGHT_POINT: return GeLoadString(IDS_LIGHT_POINT);
		}
		return ""_s;
	}

    maxon::Int num_threads;

    static maxon::Result<maxon::BaseArray<TexData const*>> FindAssignedTextureTags(BaseMaterial* mat, const InitRenderStruct &is);

    void DumpCacheToScene();
    Int64 GetCacheSize();

	Bool HandleMatPreviewGetPopupOptions(GeListNode *node, BaseContainer* bc);
	Bool HandleMatPreviewHandlePopupMsg(GeListNode *node, Int32 menu_item);
	Bool HandleMatPreviewPrepareScene(GeListNode *node, MatPreviewPrepareScene* preparescene);
	Bool HandleMatPreviewGenerateImage(GeListNode *node, MatPreviewGenerateImage* preparescene);
	Bool HandleMatPreviewModifyCacheScene(GeListNode* node, MatPreviewModifyCacheScene* cache);

  public:
    maxon::Result<void>  OnRenderingStart(VolumeData* );

    virtual	VOLUMEINFO GetRenderInfo(BaseMaterial *mat);
    virtual	INITRENDERRESULT InitRender(BaseMaterial *mat, const InitRenderStruct &irs);
    virtual	void FreeRender				(BaseMaterial *mat);

    virtual	void CalcSurface			(BaseMaterial *mat, VolumeData *vd);
    virtual	void CalcTransparency	(BaseMaterial *mat, VolumeData *vd);
    virtual	void CalcVolumetric		(BaseMaterial *mat, VolumeData *vd);
    
    virtual Bool Init(GeListNode *node);
    virtual Bool Message(GeListNode *node, Int32 type, void *data);

    virtual Bool GetDEnabling(GeListNode *node, const DescID &id,const GeData &t_data,DESCFLAGS_ENABLE flags,const BaseContainer *itemdesc);
    virtual Bool SetDParameter(GeListNode *node, const DescID &id,const GeData &t_data,DESCFLAGS_SET &flags);
    virtual Bool GetDDescription(GeListNode* node, Description* description, DESCFLAGS_DESC& flags);

    static NodeData* Alloc() { return NewObjClear(ParticleVolume); }
};


Bool ParticleVolume::Init(GeListNode *node)
{
  BaseContainer *data = ((BaseMaterial*)node)->GetDataInstance();
  if(!data) return FALSE;
  BaseContainer &b = *data;
  b.SetFloat(RADIUS_UNITS,100.0);
  b.SetFloat(RADIUS_KM,6360.0);
  b.SetFloat(GLOBAL_DENSITY,1.);
  b.SetFloat(SCATTER_INTENSITY,10.0);
  b.SetInt32(NUM_INTEGRATION_STEPS,8);
  b.SetFloat(INTEGRATION_STEP_SIZE, 100.);
  b.SetFloat(MAX_INTEGRATION_ERROR, 0.01);
  b.SetFloat(MAX_INTEGRATION_ERROR_FOR_GI, 1.);
  //b.SetInt32(MAX_SUBSTEP_LEVELS, 4);

  Int32 B = GROUP_CONSTITUENT1;
  b.SetBool(B+ID_ON,TRUE);
  b.SetFloat(B+ID_DECAY_HEIGHT,8.);
  b.SetFloat(B+ID_DENSITY,1.);
  b.SetVector(B+ID_PHASE_FACTOR,Vector(5.8,13.5,33.1));
  b.SetInt32(B+ID_PHASE_TYPE,ID_PHASE_TYPE_RAYLEIGH);
  
  B= GROUP_CONSTITUENT2;
  b.SetBool(B+ID_ON,TRUE);
  b.SetFloat(B+ID_DECAY_HEIGHT,1.2);
  b.SetFloat(B+ID_DENSITY,1.);
  b.SetVector(B+ID_PHASE_FACTOR,Vector(20.0));
  b.SetInt32(B+ID_PHASE_TYPE,ID_PHASE_TYPE_MIE);
  b.SetVector(B+ID_ABSORPTION_FACTOR, Vector(2.));

  UpdateGuiValues((BaseMaterial*)node);
  
  return TRUE;
}


void ParticleVolume::UpdateGuiValues(BaseMaterial* node)
{
  BaseContainer &b = *node->GetDataInstance();
  Float r_km = b.GetFloat(RADIUS_KM);
  Float r_units = b.GetFloat(RADIUS_UNITS);
  Float unit_p_km = r_units/r_km;
  b.SetFloat(UNIT_PER_KM,unit_p_km);
  b.SetFloat(KM_PER_UNIT,1./unit_p_km);
  
  for(Int32 grp=0; grp<NUM_CONSTITUENTS; ++grp)
  {
    Float h = b.GetFloat(ConstitId(ID_DECAY_HEIGHT,grp)) * unit_p_km;
    b.SetFloat(ConstitId(ID_DECAY_HEIGHT_UNITS,grp),h);
  }  
}


Bool ParticleVolume::SetDParameter(GeListNode *node, const DescID &id,const GeData &t_data,DESCFLAGS_SET &flags)
{
  return SUPER::SetDParameter(node,id,t_data,flags);
}


Bool ParticleVolume::GetDEnabling(GeListNode *node, const DescID &id,const GeData &t_data,DESCFLAGS_ENABLE flags,const BaseContainer *itemdesc)
{
  //if(IsReadOnly(id)) return FALSE;
  //const BaseContainer &b = *((BaseMaterial*)node)->GetDataInstance();
  Bool use_own_scaling = TRUE; //b.GetBool(USE_OWN_SCALING);
  switch(id[0].id)
  {
    case KM_PER_UNIT:
    case UNIT_PER_KM:
      return FALSE;
    case RADIUS_KM:
      return use_own_scaling;
  }
  if (id[0].id >= 2000)
  {
    Int32 cid = (id[0].id % 1000);
    switch(cid)
    {
      case ID_DECAY_HEIGHT_UNITS:
        return !use_own_scaling;
      case ID_DECAY_HEIGHT:
        return use_own_scaling;
    }
  }
  return SUPER::GetDEnabling(node,id,t_data,flags,itemdesc);
}


Bool ParticleVolume::GetDDescription(GeListNode* node, Description* description, DESCFLAGS_DESC& flags)
{
  if (!description->LoadDescription(node->GetType()))
    return false;

  flags |= DESCFLAGS_DESC::LOADED;

  const BaseContainer &b = *((BaseMaterial*)node)->GetDataInstance();
  switch (b.GetInt32(GEOMETRY_TYPE))
  {
  case ID_GEOMETRY_FLAT:
  {
	  AutoAlloc<AtomArray> arr;
	  arr->Append(static_cast<C4DAtom*>(node));

	  auto ReplaceName = [&, description](Int32 id, Int32 replacmentid)
	  {
		  BaseContainer* elem = description->GetParameterI(DescLevel(id), arr);
		  BaseContainer* replacement = description->GetParameterI(DescLevel(replacmentid), arr);
		  if (elem && replacement)
		  {
			  elem->SetString(DESC_NAME, replacement->GetString(DESC_NAME));
			  elem->SetString(DESC_SHORT_NAME, replacement->GetString(DESC_SHORT_NAME));
		  }
	  };

	  ReplaceName(RADIUS_UNITS, RADIUS_UNITS_ALTERNATIVE_NAME);
	  ReplaceName(RADIUS_KM, RADIUS_KM_ALTERNATIVE_NAME);
  }
  break;
  }

#if 0
  {
    const DescID* singleid = description->GetSingleDescID();
    const Int32 ID = DEBUG_ELEMENTS + 1;
    const DescID cid = DescLevel(ID, DTYPE_BUTTON, 0);

    if (!singleid || cid.IsPartOf(*singleid, nullptr))
    {
      BaseContainer bc = GetCustomDataTypeDefault(DTYPE_BUTTON);
      bc.SetInt32(DESC_CUSTOMGUI, CUSTOMGUI_BUTTON);
      bc.SetString(DESC_NAME, "Button"_s);
      bc.SetInt32(DESC_ANIMATE, DESC_ANIMATE_OFF);
      bc.SetInt32(DESC_SCALEH, 1);

      description->SetParameter(cid, bc, DescLevel(ID_MATERIALPROPERTIES));
    }
  }

  {
    const DescID* singleid = description->GetSingleDescID();
    const Int32 ID = DEBUG_ELEMENTS + 2;
    const DescID cid = DescLevel(ID, DTYPE_STATICTEXT, 0);

    if (!singleid || cid.IsPartOf(*singleid, nullptr))
    {
      BaseContainer bc = GetCustomDataTypeDefault(DTYPE_STATICTEXT);
      bc.SetInt32(DESC_CUSTOMGUI, CUSTOMGUI_STATICTEXT);
      bc.SetString(DESC_NAME, "# Cache Records:"_s);
      bc.SetInt32(DESC_ANIMATE, DESC_ANIMATE_OFF);
      bc.SetInt32(DESC_SCALEH, 1);

      description->SetParameter(cid, bc, DescLevel(ID_MATERIALPROPERTIES));
    }
  }
#endif
  return SUPER::GetDDescription(node, description, flags);
}


VOLUMEINFO ParticleVolume::GetRenderInfo(BaseMaterial *mat)
{
  return VOLUMEINFO::TRANSPARENCY|VOLUMEINFO::VOLUMETRIC;
}


maxon::Result<maxon::BaseArray<TexData const*>> ParticleVolume::FindAssignedTextureTags(BaseMaterial* mat, const InitRenderStruct &is)
{
  iferr_scope;
  maxon::BaseArray<TexData const *> tex_list;
  for (int i_obj = 0; i_obj < is.vd->GetObjCount(); ++i_obj)
  {
    auto *obj = is.vd->GetObj(i_obj);
    for (int i_tex = 0; i_tex < obj->texcnt; ++i_tex)
    {
      auto *tex = is.vd->GetTexData(obj, i_tex);
      if (tex->mp == mat)
      {
        tex_list.Append(tex) iferr_return;
      }
    }
  }
  return tex_list;
}


INITRENDERRESULT ParticleVolume::InitRender(BaseMaterial *mat, const InitRenderStruct &is)
{
  iferr_scope_handler {
      return INITRENDERRESULT::OUTOFMEMORY;
  };
  const BaseContainer *data = mat->GetDataInstance();  
  if(!data) return INITRENDERRESULT::UNKNOWNERROR;

#ifdef USE_RADIANCE_CACHE
  // Make shader known to the VideoPost
  auto *vp_base = is.vd->FindVideoPost(ID_ATMOSPHERE2);
  if (!vp_base) return INITRENDERRESULT::UNKNOWNERROR;
  auto *vp = vp_base->GetNodeData<AtmosphereVideopost>();
  if (!vp) return INITRENDERRESULT::UNKNOWNERROR;
  vp->shaders.Append(this) iferr_return;
#endif

  num_threads = is.vd->GetCPUCount();
  
  tex_list = FindAssignedTextureTags(mat, is) iferr_return;

  {
    included_lights.Flush();
    InExcludeData* inex = (InExcludeData*)data->GetCustomDataType(LIGHTINCLUSION,CUSTOMDATATYPE_INEXCLUDE_LIST);
    if(inex && is.vd)
    {
      InclusionTable* incl = inex->BuildInclusionTable(is.doc);
      if(incl)
      {
        included_lights.EnsureCapacity(incl->GetObjectCount()) iferr_return;
        for(Int32 i=0; i<is.vd->GetLightCount(); ++i)
        {
          const RayLight* rl = is.vd->GetLight(i);
          if(incl->Check(rl->link))
          {
            included_lights.Append(rl)  iferr_return;
          }
        }
        FreeInclusionTable(incl);
      }
    }
  }  
   
  for (Int32 j = 0; j < tex_list.GetCount(); ++j)
  {
    using RefType = decltype(renders)::ValueType;
    auto result = RefType::Create(*data, is, mat, included_lights, tex_list[j]) iferr_return;
		result->Init(*data, is) iferr_return;
    renders.Append(std::move(result))  iferr_return;
  }
  return INITRENDERRESULT::OK;
}


void ParticleVolume::FreeRender(BaseMaterial *mat)
{
  renders.Flush();
  included_lights.Flush();
}


maxon::Result<void>  ParticleVolume::OnRenderingStart(VolumeData *vd)
{
  iferr_scope;
  for (auto &r : renders)
  {
    r->OnRenderingStart(vd) iferr_return;
  }
  return maxon::OK;
}


void ParticleVolume::CalcSurface(BaseMaterial *mat, VolumeData *vd)
{
  vd->trans = Vector64(1.0);
}



void ParticleVolume::CalcVolumetric(BaseMaterial *mat, VolumeData *vd)
{
  if (vd->raydepth == 0) // If called recursively, triggered by CalcDirectLighting in this shader.
  {
    vd->trans = Vector64(1.);
    return; 
  }
  auto tex_idx = tex_list.FindIndex(vd->tex);
  if (tex_idx >= 0)
    renders[tex_idx]->CalcVolumetric(vd);
}

void ParticleVolume::CalcTransparency(BaseMaterial *mat, VolumeData *vd)
{
  auto tex_idx = tex_list.FindIndex(vd->tex);
  if (tex_idx >= 0)
    renders[tex_idx]->CalcTransparency(vd);
}


Bool ParticleVolume::HandleMatPreviewGetPopupOptions(GeListNode *node, BaseContainer* bc)
{
	const auto geom_id = GetParameterInt32(node, GEOMETRY_TYPE);
	bc->SetString(MATPREVIEW_POPUP_NAME, GeLoadString(IDS_PARTICLEVOLUME));
	switch (geom_id)
	{
	case ID_GEOMETRY_FLAT:
	{
		for (auto scene_id : { PREVIEW_ATMOS_GROUND, PREVIEW_ATMOS_GROUND2, PREVIEW_ATMOS_HIGH, PREVIEW_ATMOS_LOW })
		{
			for (auto light_id : { PREVIEW_LIGHT_DISTANT, PREVIEW_LIGHT_POINT })
			{
				const char* activation_mark = (preview_scene_id == scene_id && preview_light_type == light_id) ? "&c&" : "";
				bc->SetString(
					MakeMenuNum(scene_id, light_id, geom_id),
					ToString(scene_id) + " - "_s + ToString(light_id) + activation_mark);
			}
		}
		break;
	}
	default:
	{
		for (auto scene_id : { PREVIEW_ATMOS_GROUND, PREVIEW_ATMOS_HIGH, PREVIEW_ATMOS_LOW })
		{
			const char* activation_mark = (preview_scene_id == scene_id) ? "&c&" : "";
			bc->SetString(
				MakeMenuNum(scene_id, PREVIEW_LIGHT_DISTANT, geom_id),
				ToString(scene_id) + activation_mark);
		}
		break;
	}
	}
	return true;
}

Bool ParticleVolume::HandleMatPreviewHandlePopupMsg(GeListNode *node, Int32 menu_item)
{
	const auto geom_id = GetParameterInt32(node, GEOMETRY_TYPE);
	maxon::Tie(preview_scene_id, preview_light_type) = ExtractPreviewConfigFromMenuNum(menu_item, geom_id);
	return true;
}


Bool ParticleVolume::HandleMatPreviewModifyCacheScene(GeListNode* node, MatPreviewModifyCacheScene* cache)
{
	const auto geom_id = GetParameterInt32(node, GEOMETRY_TYPE);
	if (!cache->pDoc)
		return false;

	return true;
}


Bool ParticleVolume::HandleMatPreviewPrepareScene(GeListNode *node, MatPreviewPrepareScene* preparescene)
{
	const auto geom_id = GetParameterInt32(node, GEOMETRY_TYPE);
	auto* doc = preparescene->pDoc;


	// Load the scene
	auto path = GeGetPluginPath() + String("res") + String("scene");
	switch (geom_id)
	{
	case ID_GEOMETRY_FLAT:
		path += "Atmosphere Flat.c4d"_s;
		break;
	case ID_GEOMETRY_CYLINDER:
		switch (preview_scene_id)
		{
		case PREVIEW_ATMOS_GROUND:
			path += "Atmosphere Ring Ground.c4d"_s;
			break;
		case PREVIEW_ATMOS_LOW:
			path += "Atmosphere Ring Low.c4d"_s;
			break;
		default:
			path += "Atmosphere Ring High.c4d"_s;
			break;
		}
		break;
	default:
		switch (preview_scene_id)
		{
		case PREVIEW_ATMOS_GROUND:
			path += "Atmosphere Ground.c4d"_s;
			break;
		case PREVIEW_ATMOS_LOW:
			path += "Atmosphere Low.c4d"_s;
			break;
		default:
			path += "Atmosphere Preview.c4d"_s;
			break;
		}
	}
	auto loadflags =
		SCENEFILTER::NOUNDO |
		SCENEFILTER::MERGESCENE |
		SCENEFILTER::OBJECTS |
		SCENEFILTER::MATERIALS;
	if (!MergeDocument(doc, path, loadflags, nullptr))
		return false;


	// Copy self into the scene
	AutoAlloc<AliasTrans> trans;
	if (!trans)
		return FALSE;
	if (!trans->Init(GetActiveDocument()))
		return FALSE;
	BaseMaterial* matclone = (BaseMaterial*)(Get()->GetClone(COPYFLAGS::NONE, trans));
	doc->InsertMaterial(matclone);
	trans->Translate(TRUE);
	if (preparescene->pLink) preparescene->pLink->SetLink(matclone); // necessary
	const Float radius_in_preview_scene = 100.;
	matclone->SetParameter(RADIUS_UNITS, GeData(radius_in_preview_scene), DESCFLAGS_SET::NONE);
	matclone->SetParameter(LIGHTINCLUSION, GeData(DA_CUSTOMDATATYPE, DEFAULTVALUE), DESCFLAGS_SET::NONE);

	BaseObject* atmosphere = nullptr;

	// Prepare the scene
	if (geom_id == ID_GEOMETRY_FLAT)
	{
		auto* light = doc->SearchObject(
			preview_light_type == PREVIEW_LIGHT_DISTANT ?
			"Object_DirectionalLight"_s :
			"Light_Point"_s);
		assert(light);
		if (!light)
			return false;
		light->SetParameter(ID_BASEOBJECT_VISIBILITY_EDITOR, GeData(OBJECT_ON), DESCFLAGS_SET::NONE);
		light->SetParameter(ID_BASEOBJECT_VISIBILITY_RENDER, GeData(OBJECT_ON), DESCFLAGS_SET::NONE);
		
		//if (preview_light_type == PREVIEW_LIGHT_DISTANT)
		//	light->SetName("Object"_s);
		
		BaseObject* camera_holder = nullptr;
		switch (preview_scene_id)
		{
		case PREVIEW_ATMOS_GROUND: camera_holder = doc->SearchObject("Object_CamGround"_s); break;
		case PREVIEW_ATMOS_GROUND2: camera_holder = doc->SearchObject("Object_CamGroundCloser"_s); break;
		case PREVIEW_ATMOS_HIGH: camera_holder = doc->SearchObject("Object_CamHigh"_s); break;
		case PREVIEW_ATMOS_LOW: camera_holder = doc->SearchObject("Object_CamLow"_s); break;
		}
		assert(camera_holder);
		if (!camera_holder)
			return false;
		
		//if (preview_light_type == PREVIEW_LIGHT_POINT)
		//	camera_holder->SetName("Object"_s);

		auto* camera = FindObject(camera_holder, "Camera"_s);
		assert(camera);
		if (!camera)
			return false;
		doc->GetRenderBaseDraw()->SetSceneCamera(camera);

		atmosphere = doc->SearchObject("Atmosphere"_s);
		assert(atmosphere);
	}
	else if (geom_id == ID_GEOMETRY_CYLINDER)
	{
		// Setup camera
		auto *cam = doc->SearchObject("Camera"_s);
		assert(cam);
		if (!cam)
			return false;
		doc->GetRenderBaseDraw()->SetSceneCamera(cam);

		if (preview_scene_id == PREVIEW_ATMOS_GROUND)
			matclone->SetParameter(RADIUS_UNITS, GeData(1000.), DESCFLAGS_SET::NONE);

		// Link material in the texture tag
		atmosphere = doc->SearchObject("Atmosphere"_s);
		assert(atmosphere);
	}
	else
	{
		// Setup camera
		auto *cam = doc->SearchObject("Camera"_s);
		if (!cam)
			return false;
		doc->GetRenderBaseDraw()->SetSceneCamera(cam);

		// Link material in the texture tag
		atmosphere = doc->SearchObject("Object"_s);
		assert(atmosphere);
	}

	if (!atmosphere)
		return false;
	{
		TextureTag* textag = (TextureTag*)atmosphere->GetTag(Ttexture);
		if (textag)
			textag->SetMaterial(matclone);
	}

	//SaveDocument(doc, "D:\\shitfuck.c4d", SAVEDOCUMENTFLAGS::NONE, FORMAT_C4DEXPORT);
	preparescene->bScenePrepared = TRUE; // inform the preview that the scene is prepared now
	return true;
}

Bool ParticleVolume::HandleMatPreviewGenerateImage(GeListNode *node, MatPreviewGenerateImage* image)
{
	if (image->pDoc)
	{
		if (!image->bEditorPreview)
		{
			// we don't calculate a preview map for the editor
			Int32 w = image->pDest->GetBw();
			Int32 h = image->pDest->GetBh();
			BaseContainer bcRender = image->pDoc->GetActiveRenderData()->GetData();
			bcRender.SetInt32(RDATA_XRES, w);
			bcRender.SetInt32(RDATA_YRES, h);
			bcRender.SetInt32(RDATA_ANTIALIASING, ANTI_GEOMETRY);
			if (image->bLowQuality)
#if API_VERSION < 21000
				bcRender.SetInt32(RDATA_RENDERENGINE, RDATA_RENDERENGINE_PREVIEWSOFTWARE);
#else
				bcRender.SetInt32(RDATA_RENDERENGINE, RDATA_RENDERENGINE_PREVIEWHARDWARE);
#endif
			image->pDest->Clear(0, 0, 0);
			image->lResult = RenderDocument(image->pDoc, bcRender, NULL, NULL, image->pDest,
				RENDERFLAGS::EXTERNAL | RENDERFLAGS::PREVIEWRENDER, image->pThread);
		}
	}
	return true;
}



Bool ParticleVolume::Message(GeListNode *node, Int32 type, void *data)
{
  switch (type)
  {
    case MSG_DESCRIPTION_VALIDATE:
    {
      UpdateGuiValues((BaseMaterial*)node);
#ifdef USE_RADIANCE_CACHE
      { // Update Debug Fields
        Int64 num_cr = 0;
        for (auto &render : renders)
          num_cr += render->GetCacheSize();
        node->SetParameter(DescID(DEBUG_ELEMENTS + 2), GeData(maxon::ToString(num_cr, nullptr)), DESCFLAGS_SET::NONE);
      }
#endif
	  return TRUE;
    }
    case MATPREVIEW_GET_OBJECT_INFO:
    {
      MatPreviewObjectInfo* info = (MatPreviewObjectInfo*)data;
      info->bHandlePreview = TRUE; // own preview handling
      info->bNeedsOwnScene = TRUE;
      info->bNoStandardScene = true;
      info->lFlags = MATPREVIEW_FLAG_HIDE_ROTATION; //MATPREVIEW_FLAG_HIDE_SCENES; //MATPREVIEW_FLAG_HIDE_SCENE_SETTINGS;
    }
    return TRUE;
    case MATPREVIEW_GET_POPUP_OPTIONS:
    {
		BaseContainer* bc = (BaseContainer*)data;
		return HandleMatPreviewGetPopupOptions(node, bc);
    }
    case MATPREVIEW_HANDLE_POPUP_MSG:
    {
      Int32 l = *((Int32*)data);
	  return HandleMatPreviewHandlePopupMsg(node, l);
    }
    case MATPREVIEW_MODIFY_CACHE_SCENE:
    {
      MatPreviewModifyCacheScene* scene = (MatPreviewModifyCacheScene*)data;
	  if (!scene)
		  return false;
	  return HandleMatPreviewModifyCacheScene(node, scene);
    }
    case MATPREVIEW_PREPARE_SCENE:
    {
      auto* preparescene = static_cast<MatPreviewPrepareScene*>(data);
	  return HandleMatPreviewPrepareScene(node, preparescene);
    }
    case MATPREVIEW_GENERATE_IMAGE:
    {
      MatPreviewGenerateImage* image = (MatPreviewGenerateImage*)data;
	  return HandleMatPreviewGenerateImage(node, image);
    }
    case MSG_DESCRIPTION_COMMAND:
    {
      DescriptionCommand* dc = (DescriptionCommand*)data;

      const Int32 id = dc->_descId[0].id;

      switch (id)
      {
        case DEBUG_ELEMENTS + 1:
        {
          DumpCacheToScene();
        }
        break;
      }
    }
  }
  return MaterialData::Message(node, type, data);
}



void ParticleVolume::DumpCacheToScene()
{
  auto* doc = GetActiveDocument();
  for (auto &render : renders)
  {
    render->DumpCacheTo(doc);
  }
  EventAdd(EVENT::FORCEREDRAW);
}

#if 0
Bool AtmosphereVideopost::Init(GeListNode* node)
{
  return true;
}

Bool AtmosphereVideopost::RenderEngineCheck(BaseVideoPost* node, Int32 id)
{
  // the following render engines are not supported by this effect
  if (id == RDATA_RENDERENGINE_PREVIEWSOFTWARE ||
    id == RDATA_RENDERENGINE_PREVIEWHARDWARE ||
    id == RDATA_RENDERENGINE_CINEMAN)
    return false;

  return true;
}

RENDERRESULT AtmosphereVideopost::Execute(BaseVideoPost* node, VideoPostStruct* vps)
{
  iferr_scope_handler{
    return RENDERRESULT::OUTOFMEMORY;
  };
  if (vps->vp == VIDEOPOSTCALL::INNER)
  {
    for (auto *shader : shaders)
    {
      shader->OnRenderingStart(vps->vd) iferr_return;
    }
  }

  return RENDERRESULT::OK;
}
#endif


Bool RegisterShader(void)
{
  // decide by name if the plugin shall be registered - just for user convenience
  String name = GeLoadString(IDS_PARTICLEVOLUME); if (name.IsEmpty()) return TRUE;
  name = GeGetDefaultFilename(DEFAULTFILENAME_SHADER_VOLUME) + name; // place in default Shader section
  if (!RegisterMaterialPlugin(ID_PARTICLEVOLUME, name, 0, ParticleVolume::Alloc, "Mplanetatmosphere"_s, 0)) return FALSE;
  return TRUE;
#ifdef USE_RADIANCE_CACHE
  return RegisterVideoPostPlugin(ID_ATMOSPHERE2, "AtmosphereVP"_s, PLUGINFLAG_VIDEOPOST_MULTIPLE, AtmosphereVideopost::Alloc, "VPhairsdkpost"_s, 0, 0);
#endif
}


#pragma endregion

}


#if 0
template<class T, int N>
struct StaticVector
{
	using FixedBufferArray =

		char buffer[N * sizeof(T)];
	int count = 0;

	T* Data()
	{
		return std::launder(reinterpret_cast<T*>(buffer));
	}

	~StaticVector()
	{
		std::destroy_n(Data(), count);
	}

	void PushBack(const T& elem)
	{
		std::uninitialized
			values[count] = elem;
	}

	T& operator[](int i) {
		return values[i];
	}

	const T& operator[](int i) const {
		return values[i];
	}

};
#endif
